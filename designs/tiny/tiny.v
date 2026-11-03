module tiny(a, b, clk, y, q);
  input a, b, clk; output y, q;
  wire n1, n2;
  sky130_fd_sc_hd__inv_1 u1 (.A(a), .Y(n1));
  sky130_fd_sc_hd__nand2_1 u2 (.A(n1), .B(b), .Y(n2));
  sky130_fd_sc_hd__inv_1 u3 (.A(n2), .Y(y));
  sky130_fd_sc_hd__dfxtp_1 r1 (.CLK(clk), .D(y), .Q(q));
endmodule
