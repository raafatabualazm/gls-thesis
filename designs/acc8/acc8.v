// small open test design (written for this repo): 8-bit accumulator/ALU with registered output
module acc8(input clk, input rst, input [1:0] op, input [7:0] a, output reg [7:0] acc, output zero, output parity);
  always @(posedge clk)
    if (rst) acc <= 8'd0;
    else case (op)
      2'd0: acc <= acc + a;
      2'd1: acc <= acc ^ a;
      2'd2: acc <= acc & ~a;
      2'd3: acc <= {acc[6:0], acc[7]} - a;
    endcase
  assign zero = (acc == 0);
  assign parity = ^acc;
endmodule
