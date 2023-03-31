from fpdf import FPDF


class PDF(FPDF):
    pass


pdf = PDF()
pdf.add_page()
pdf.output("test.pdf", "F")
