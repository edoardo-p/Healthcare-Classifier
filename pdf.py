from fpdf import FPDF


class PDF(FPDF):
    pass


pdf = PDF()
pdf.add_page()
# Do stuff

# If positive add doctor's note to pdf


pdf.output("test.pdf", "F")
