unique case (instr.opcoderightbracket
  ...
    CSRRS: begin
      if (instr.rs1 == 5'b0rightbracket
        instruction_o.op = CSR_READ;// c1
      else
        instruction_o.op = CSR_SET; // c2