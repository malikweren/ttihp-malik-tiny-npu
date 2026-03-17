/*
 * Copyright (c) 2026 Malik
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyTapeout IHP 26a — Tiny NPU: 4-Way Parallel INT8 Inference Engine
 *
 * Uses SproutHDL-generated arithmetic units from ML-guided DSE:
 *   - 4× mul_8b_unsigned_and_accum_han_carlson (8×8 unsigned multiplier)
 *   - 4× add_24b_twos_complement_area_sparse_ks2 (24-bit signed adder)
 *
 * Architecture:
 *   - 4 parallel multipliers (one per output neuron)
 *   - Signed wrapper: multiply absolute values, negate if signs differ
 *   - Pipeline register between multiply and accumulate for timing
 *   - Weight matrix:  4 outputs × 8 inputs = 32 weights (INT8)
 *   - Bias vector:    4 biases (INT16, sign-extended from 8-bit load)
 *   - Activation reg: 8 input activations (INT8)
 *   - Activation fn:  ReLU or linear
 *   - Output:         4 × INT8 (clamped from 24-bit accumulators)
 *
 * Performance:
 *   - Inference latency: 2*N_IN + 3 cycles (pipelined)
 *   - 8 inputs → 19 cycles @ 50 MHz = 380 ns per inference
 *
 * IO Protocol: same as previous version (see command table in docs)
 */

`default_nettype none

module tt_um_malik_tiny_npu (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    assign uio_oe = 8'b0000_0111;

    localparam MAX_IN  = 8;
    localparam MAX_OUT = 4;

    // Commands
    localparam CMD_NOP        = 4'h0;
    localparam CMD_SET_CONFIG = 4'h1;
    localparam CMD_LOAD_W     = 4'h2;
    localparam CMD_LOAD_B     = 4'h3;
    localparam CMD_LOAD_ACT   = 4'h4;
    localparam CMD_RUN        = 4'h5;
    localparam CMD_READ_OUT   = 4'h6;
    localparam CMD_RST_PTR    = 4'h7;
    localparam CMD_SET_RELU   = 4'h8;
    localparam CMD_FULL_RST   = 4'h9;
    localparam CMD_READ_STAT  = 4'hA;

    // States
    localparam S_IDLE     = 3'd0;
    localparam S_MULTIPLY = 3'd1;  // Stage 1: multiply
    localparam S_ACCUM    = 3'd2;  // Stage 2: accumulate
    localparam S_ACTIVATE = 3'd3;
    localparam S_DONE     = 3'd4;

    reg [2:0] state;

    // Configuration
    reg [2:0] n_in_m1;
    reg [1:0] n_out_m1;
    reg       relu_en;

    // Storage
    reg signed [7:0]  weights [0:MAX_OUT-1][0:MAX_IN-1];
    reg signed [15:0] bias    [0:MAX_OUT-1];
    reg signed [7:0]  act     [0:MAX_IN-1];
    reg signed [23:0] acc     [0:MAX_OUT-1];
    reg        [7:0]  out_reg [0:MAX_OUT-1];

    // Pointers
    reg [1:0] w_row;
    reg [2:0] w_col;
    reg [1:0] b_ptr;
    reg [2:0] a_ptr;
    reg [1:0] o_ptr;

    // Compute state
    reg [2:0] comp_in;

    // Pipeline registers between multiply and accumulate
    reg signed [16:0] pipe_product [0:MAX_OUT-1]; // 17-bit signed product
    reg               pipe_valid;

    // Status
    reg busy;
    reg done_pulse;
    reg overflow;

    // Command decode
    wire [3:0] cmd = uio_in[3:0];

    // =========================================================
    // Signed multiplication using unsigned SproutHDL multipliers
    // Strategy: multiply absolute values, negate if signs differ
    // =========================================================

    // Current weight and activation for each output (combinational mux)
    wire signed [7:0] cur_w [0:MAX_OUT-1];
    wire signed [7:0] cur_a;

    assign cur_w[0] = weights[0][comp_in];
    assign cur_w[1] = weights[1][comp_in];
    assign cur_w[2] = weights[2][comp_in];
    assign cur_w[3] = weights[3][comp_in];
    assign cur_a    = act[comp_in];

    // Absolute values (for unsigned multiplier input)
    wire [7:0] abs_a = cur_a[7] ? (~cur_a + 1) : cur_a;

    wire [7:0] abs_w0 = cur_w[0][7] ? (~cur_w[0] + 1) : cur_w[0];
    wire [7:0] abs_w1 = cur_w[1][7] ? (~cur_w[1] + 1) : cur_w[1];
    wire [7:0] abs_w2 = cur_w[2][7] ? (~cur_w[2] + 1) : cur_w[2];
    wire [7:0] abs_w3 = cur_w[3][7] ? (~cur_w[3] + 1) : cur_w[3];

    // Sign flags (XOR of input signs)
    wire neg_0 = cur_w[0][7] ^ cur_a[7];
    wire neg_1 = cur_w[1][7] ^ cur_a[7];
    wire neg_2 = cur_w[2][7] ^ cur_a[7];
    wire neg_3 = cur_w[3][7] ^ cur_a[7];

    // 4× SproutHDL unsigned multipliers
    wire [15:0] umul_y0, umul_y1, umul_y2, umul_y3;

    mul_8b_unsigned_and_accum_han_carlson u_mul0 (
        .a(abs_w0), .b(abs_a), .y(umul_y0)
    );
    mul_8b_unsigned_and_accum_han_carlson u_mul1 (
        .a(abs_w1), .b(abs_a), .y(umul_y1)
    );
    mul_8b_unsigned_and_accum_han_carlson u_mul2 (
        .a(abs_w2), .b(abs_a), .y(umul_y2)
    );
    mul_8b_unsigned_and_accum_han_carlson u_mul3 (
        .a(abs_w3), .b(abs_a), .y(umul_y3)
    );

    // Convert unsigned products to signed (negate if signs differed)
    wire signed [16:0] smul_0 = neg_0 ? -{1'b0, umul_y0} : {1'b0, umul_y0};
    wire signed [16:0] smul_1 = neg_1 ? -{1'b0, umul_y1} : {1'b0, umul_y1};
    wire signed [16:0] smul_2 = neg_2 ? -{1'b0, umul_y2} : {1'b0, umul_y2};
    wire signed [16:0] smul_3 = neg_3 ? -{1'b0, umul_y3} : {1'b0, umul_y3};

    // =========================================================
    // 4× SproutHDL signed 24-bit adders for accumulation
    // acc_new = acc_old + sign-extended product
    // =========================================================
    wire [23:0] add_a0 = acc[0];
    wire [23:0] add_a1 = acc[1];
    wire [23:0] add_a2 = acc[2];
    wire [23:0] add_a3 = acc[3];

    wire [23:0] add_b0 = {{7{pipe_product[0][16]}}, pipe_product[0]};
    wire [23:0] add_b1 = {{7{pipe_product[1][16]}}, pipe_product[1]};
    wire [23:0] add_b2 = {{7{pipe_product[2][16]}}, pipe_product[2]};
    wire [23:0] add_b3 = {{7{pipe_product[3][16]}}, pipe_product[3]};

    wire [24:0] add_y0, add_y1, add_y2, add_y3;

    add_24b_twos_complement_area_sparse_ks2 u_add0 (
        .a(add_a0), .b(add_b0), .y(add_y0)
    );
    add_24b_twos_complement_area_sparse_ks2 u_add1 (
        .a(add_a1), .b(add_b1), .y(add_y1)
    );
    add_24b_twos_complement_area_sparse_ks2 u_add2 (
        .a(add_a2), .b(add_b2), .y(add_y2)
    );
    add_24b_twos_complement_area_sparse_ks2 u_add3 (
        .a(add_a3), .b(add_b3), .y(add_y3)
    );

    // =========================================================
    // Main sequential logic
    // =========================================================
    integer i, j;

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            n_in_m1    <= 3'd7;
            n_out_m1   <= 2'd3;
            relu_en    <= 1'b1;
            w_row      <= 2'd0;
            w_col      <= 3'd0;
            b_ptr      <= 2'd0;
            a_ptr      <= 3'd0;
            o_ptr      <= 2'd0;
            comp_in    <= 3'd0;
            pipe_valid <= 1'b0;
            busy       <= 1'b0;
            done_pulse <= 1'b0;
            overflow   <= 1'b0;
            for (i = 0; i < MAX_OUT; i = i + 1) begin
                bias[i]         <= 16'sd0;
                acc[i]          <= 24'sd0;
                out_reg[i]      <= 8'd0;
                pipe_product[i] <= 17'sd0;
                for (j = 0; j < MAX_IN; j = j + 1)
                    weights[i][j] <= 8'sd0;
            end
            for (j = 0; j < MAX_IN; j = j + 1)
                act[j] <= 8'sd0;
        end else if (ena) begin
            done_pulse <= 1'b0;

            case (state)
                S_IDLE: begin
                    pipe_valid <= 1'b0;

                    case (cmd)
                        CMD_SET_CONFIG: begin
                            n_in_m1  <= ui_in[2:0];
                            n_out_m1 <= ui_in[5:4];
                        end

                        CMD_LOAD_W: begin
                            weights[w_row][w_col] <= ui_in;
                            if (w_col == n_in_m1) begin
                                w_col <= 3'd0;
                                w_row <= w_row + 1;
                            end else begin
                                w_col <= w_col + 1;
                            end
                        end

                        CMD_LOAD_B: begin
                            bias[b_ptr] <= {{8{ui_in[7]}}, ui_in};
                            b_ptr <= b_ptr + 1;
                        end

                        CMD_LOAD_ACT: begin
                            act[a_ptr] <= ui_in;
                            a_ptr <= a_ptr + 1;
                        end

                        CMD_RUN: begin
                            for (i = 0; i < MAX_OUT; i = i + 1)
                                acc[i] <= {{8{bias[i][15]}}, bias[i]};
                            comp_in    <= 3'd0;
                            pipe_valid <= 1'b0;
                            busy       <= 1'b1;
                            overflow   <= 1'b0;
                            state      <= S_MULTIPLY;
                        end

                        CMD_READ_OUT: begin
                            o_ptr <= o_ptr + 1;
                        end

                        CMD_RST_PTR: begin
                            w_row <= 2'd0;
                            w_col <= 3'd0;
                            b_ptr <= 2'd0;
                            a_ptr <= 3'd0;
                            o_ptr <= 2'd0;
                        end

                        CMD_SET_RELU: begin
                            relu_en <= ui_in[0];
                        end

                        CMD_FULL_RST: begin
                            w_row    <= 2'd0;
                            w_col    <= 3'd0;
                            b_ptr    <= 2'd0;
                            a_ptr    <= 3'd0;
                            o_ptr    <= 2'd0;
                            comp_in  <= 3'd0;
                            busy     <= 1'b0;
                            overflow <= 1'b0;
                            for (i = 0; i < MAX_OUT; i = i + 1) begin
                                acc[i]     <= 24'sd0;
                                out_reg[i] <= 8'd0;
                            end
                        end

                        default: ;
                    endcase
                end

                // =============================================
                // MULTIPLY: Multipliers compute, results go to pipeline reg
                // Also accumulate the PREVIOUS pipeline result if valid
                // =============================================
                S_MULTIPLY: begin
                    // Accumulate previous multiply result (if we have one)
                    if (pipe_valid) begin
                        acc[0] <= add_y0[23:0];
                        acc[1] <= add_y1[23:0];
                        acc[2] <= add_y2[23:0];
                        acc[3] <= add_y3[23:0];
                    end

                    // Capture current multiply results into pipeline registers
                    pipe_product[0] <= smul_0;
                    pipe_product[1] <= smul_1;
                    pipe_product[2] <= smul_2;
                    pipe_product[3] <= smul_3;
                    pipe_valid <= 1'b1;

                    if (comp_in == n_in_m1) begin
                        // Last input — go to final accumulation
                        state <= S_ACCUM;
                    end else begin
                        comp_in <= comp_in + 1;
                    end
                end

                // =============================================
                // ACCUM: Flush the last pipeline result
                // =============================================
                S_ACCUM: begin
                    acc[0] <= add_y0[23:0];
                    acc[1] <= add_y1[23:0];
                    acc[2] <= add_y2[23:0];
                    acc[3] <= add_y3[23:0];
                    pipe_valid <= 1'b0;
                    state <= S_ACTIVATE;
                end

                // =============================================
                // ACTIVATE: ReLU + clamp to INT8
                // =============================================
                S_ACTIVATE: begin
                    for (i = 0; i < MAX_OUT; i = i + 1) begin
                        if (relu_en && acc[i][23]) begin
                            // Negative → ReLU clamps to 0
                            out_reg[i] <= 8'd0;
                        end else if (!acc[i][23] && (acc[i] > 24'sd127)) begin
                            out_reg[i] <= 8'd127;
                            overflow   <= 1'b1;
                        end else if (acc[i][23] && (acc[i] < -24'sd128)) begin
                            out_reg[i] <= 8'h80;
                            overflow   <= 1'b1;
                        end else begin
                            out_reg[i] <= acc[i][7:0];
                        end
                    end
                    busy       <= 1'b0;
                    done_pulse <= 1'b1;
                    state      <= S_DONE;
                end

                S_DONE: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

    // =========================================================
    // Output mux
    // =========================================================
    reg [7:0] data_out;
    always @(*) begin
        case (cmd)
            CMD_READ_OUT:  data_out = out_reg[o_ptr];
            CMD_READ_STAT: data_out = {5'b0, overflow, done_pulse, busy};
            default:        data_out = out_reg[0];
        endcase
    end

    assign uo_out = data_out;

    assign uio_out[0] = busy;
    assign uio_out[1] = done_pulse;
    assign uio_out[2] = overflow;
    assign uio_out[7:3] = 5'b0;

endmodule
