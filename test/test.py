"""
Cocotb testbench for tt_um_malik_tiny_npu
4-way parallel INT8 inference engine with SproutHDL arithmetic units

Pipelined: multiply stage → accumulate stage
Compute latency: 2*N_IN + 3 cycles (pipeline fill + flush)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

CMD_NOP        = 0x0
CMD_SET_CONFIG = 0x1
CMD_LOAD_W     = 0x2
CMD_LOAD_B     = 0x3
CMD_LOAD_ACT   = 0x4
CMD_RUN        = 0x5
CMD_READ_OUT   = 0x6
CMD_RST_PTR    = 0x7
CMD_SET_RELU   = 0x8
CMD_FULL_RST   = 0x9
CMD_READ_STAT  = 0xA


def to_u8(val):
    return val & 0xFF


def from_s8(val):
    if val & 0x80:
        return val - 256
    return val


async def send_cmd(dut, cmd, data=0):
    dut.ui_in.value = to_u8(data)
    dut.uio_in.value = cmd & 0xF
    await ClockCycles(dut.clk, 1)


async def nop(dut, cycles=1):
    dut.uio_in.value = CMD_NOP
    dut.ui_in.value = 0
    await ClockCycles(dut.clk, cycles)


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def set_config(dut, n_in, n_out):
    data = ((n_in - 1) & 0x7) | (((n_out - 1) & 0x3) << 4)
    await send_cmd(dut, CMD_SET_CONFIG, data)
    await nop(dut)


async def load_weights_rowmajor(dut, weight_matrix):
    await send_cmd(dut, CMD_RST_PTR)
    await nop(dut)
    for row in weight_matrix:
        for w in row:
            await send_cmd(dut, CMD_LOAD_W, w)
    await nop(dut)


async def load_biases(dut, biases):
    for b in biases:
        await send_cmd(dut, CMD_LOAD_B, b)
    await nop(dut)


async def load_activations(dut, activations):
    for a in activations:
        await send_cmd(dut, CMD_LOAD_ACT, a)
    await nop(dut)


async def run_inference(dut, n_in):
    """Trigger inference. Pipelined: ~2*N_IN + 3 cycles."""
    await send_cmd(dut, CMD_RUN)
    await nop(dut)
    max_cycles = 2 * n_in + 20
    for _ in range(max_cycles):
        await ClockCycles(dut.clk, 1)
        busy = int(dut.uio_out.value) & 0x1
        if not busy:
            break
    await nop(dut, 2)


async def read_outputs(dut, n_out):
    await send_cmd(dut, CMD_RST_PTR)
    await nop(dut)
    results = []
    for _ in range(n_out):
        await send_cmd(dut, CMD_READ_OUT)
        val = int(dut.uo_out.value)
        results.append(val)
    await nop(dut)
    return results


@cocotb.test()
async def test_reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    outputs = await read_outputs(dut, 4)
    for i, v in enumerate(outputs):
        assert v == 0, f"Output {i} not zero after reset: {v}"
    dut._log.info("PASS: reset")


@cocotb.test()
async def test_simple_no_relu(dut):
    """W = [[3, 5]], x = [2, 4] → y = 6+20 = 26"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=2, n_out=1)
    await send_cmd(dut, CMD_SET_RELU, 0)
    await nop(dut)

    await load_weights_rowmajor(dut, [[3, 5]])
    await load_biases(dut, [0])
    await load_activations(dut, [2, 4])
    await run_inference(dut, 2)
    results = await read_outputs(dut, 1)

    assert results[0] == 26, f"Expected 26, got {results[0]}"
    dut._log.info("PASS: 2→1 = 26")


@cocotb.test()
async def test_with_bias(dut):
    """W = [[1, 1]], b = [10], x = [3, 7] → 3+7+10 = 20"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=2, n_out=1)
    await send_cmd(dut, CMD_SET_RELU, 0)
    await nop(dut)

    await load_weights_rowmajor(dut, [[1, 1]])
    await load_biases(dut, [10])
    await load_activations(dut, [3, 7])
    await run_inference(dut, 2)
    results = await read_outputs(dut, 1)

    assert results[0] == 20, f"Expected 20, got {results[0]}"
    dut._log.info("PASS: with bias = 20")


@cocotb.test()
async def test_relu_parallel(dut):
    """W = [[1,1],[-2,-3]], x = [5,3]
    y0 = 8 → ReLU → 8
    y1 = -19 → ReLU → 0
    Both computed in parallel.
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=2, n_out=2)
    await send_cmd(dut, CMD_SET_RELU, 1)
    await nop(dut)

    await load_weights_rowmajor(dut, [[1, 1], [-2, -3]])
    await load_biases(dut, [0, 0])
    await load_activations(dut, [5, 3])
    await run_inference(dut, 2)
    results = await read_outputs(dut, 2)

    assert results[0] == 8, f"y0: expected 8, got {results[0]}"
    assert results[1] == 0, f"y1: expected 0, got {results[1]}"
    dut._log.info("PASS: ReLU parallel")


@cocotb.test()
async def test_4x4_identity(dut):
    """Identity matrix: y = x"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=4, n_out=4)
    await send_cmd(dut, CMD_SET_RELU, 0)
    await nop(dut)

    await load_weights_rowmajor(dut, [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    await load_biases(dut, [0, 0, 0, 0])
    await load_activations(dut, [10, 20, 30, 40])
    await run_inference(dut, 4)
    results = await read_outputs(dut, 4)

    expected = [10, 20, 30, 40]
    for i in range(4):
        assert results[i] == expected[i], \
            f"y{i}: expected {expected[i]}, got {results[i]}"
    dut._log.info("PASS: 4×4 identity")


@cocotb.test()
async def test_full_4x2(dut):
    """W = [[1,2,3,4],[1,0,-1,0]], b = [0,5], x = [1,2,3,4]
    y0 = 1+4+9+16 = 30
    y1 = 1+0-3+0+5 = 3
    """
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=4, n_out=2)
    await send_cmd(dut, CMD_SET_RELU, 1)
    await nop(dut)

    await load_weights_rowmajor(dut, [[1, 2, 3, 4], [1, 0, -1, 0]])
    await load_biases(dut, [0, 5])
    await load_activations(dut, [1, 2, 3, 4])
    await run_inference(dut, 4)
    results = await read_outputs(dut, 2)

    assert results[0] == 30, f"y0: expected 30, got {results[0]}"
    assert results[1] == 3,  f"y1: expected 3, got {results[1]}"
    dut._log.info("PASS: 4→2 layer")


@cocotb.test()
async def test_signed_negative(dut):
    """W = [[-5]], x = [10] → -50 → no ReLU → 0xCE"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=1, n_out=1)
    await send_cmd(dut, CMD_SET_RELU, 0)
    await nop(dut)

    await load_weights_rowmajor(dut, [[-5]])
    await load_biases(dut, [0])
    await load_activations(dut, [10])
    await run_inference(dut, 1)
    results = await read_outputs(dut, 1)

    expected = (-50) & 0xFF
    assert results[0] == expected, f"Expected {expected}, got {results[0]}"
    dut._log.info(f"PASS: signed = {from_s8(results[0])}")


@cocotb.test()
async def test_saturation(dut):
    """W = [[127]], x = [2] → 254 → clamp to 127"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=1, n_out=1)
    await send_cmd(dut, CMD_SET_RELU, 1)
    await nop(dut)

    await load_weights_rowmajor(dut, [[127]])
    await load_biases(dut, [0])
    await load_activations(dut, [2])
    await run_inference(dut, 1)
    results = await read_outputs(dut, 1)

    assert results[0] == 127, f"Expected 127, got {results[0]}"
    dut._log.info("PASS: saturation")


@cocotb.test()
async def test_weight_reuse(dut):
    """Weights persist across inferences."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=2, n_out=1)
    await send_cmd(dut, CMD_SET_RELU, 0)
    await nop(dut)

    await load_weights_rowmajor(dut, [[2, 3]])
    await load_biases(dut, [0])

    await load_activations(dut, [1, 1])
    await run_inference(dut, 2)
    r1 = await read_outputs(dut, 1)
    assert r1[0] == 5, f"Run 1: expected 5, got {r1[0]}"

    await send_cmd(dut, CMD_RST_PTR)
    await nop(dut)
    await load_activations(dut, [10, 20])
    await run_inference(dut, 2)
    r2 = await read_outputs(dut, 1)
    assert r2[0] == 80, f"Run 2: expected 80, got {r2[0]}"

    dut._log.info("PASS: weight reuse")


@cocotb.test()
async def test_full_reset(dut):
    """CMD_FULL_RST zeros outputs."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await set_config(dut, n_in=1, n_out=1)
    await send_cmd(dut, CMD_SET_RELU, 0)
    await nop(dut)
    await load_weights_rowmajor(dut, [[10]])
    await load_biases(dut, [0])
    await load_activations(dut, [5])
    await run_inference(dut, 1)
    r = await read_outputs(dut, 1)
    assert r[0] == 50

    await send_cmd(dut, CMD_FULL_RST)
    await nop(dut)
    r = await read_outputs(dut, 1)
    assert r[0] == 0, f"After reset: expected 0, got {r[0]}"
    dut._log.info("PASS: full reset")
