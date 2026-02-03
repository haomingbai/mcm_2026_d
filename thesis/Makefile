# Makefile for compiling multiple LaTeX documents and merging PDFs
# Usage:
#   make            # build all PDFs and merge
#   make merge      # merge PDFs only (will build prerequisites)
#   make clean      # remove build artifacts and generated PDFs

LATEXMK ?= latexmk
PDFUNITE ?= pdfunite

LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

TEX_SRCS := mcm_thesis_work_1.tex letterwork.tex mcm_ai_report.tex
PDFS := $(TEX_SRCS:.tex=.pdf)

MERGED_PDF := combined.pdf

.PHONY: all compile merge clean veryclean check-tools

all: merge

compile: $(PDFS)

merge: check-tools $(MERGED_PDF)

$(MERGED_PDF): $(PDFS)
	$(PDFUNITE) $(PDFS) $@

%.pdf: %.tex
	$(LATEXMK) $(LATEXMK_FLAGS) $<

check-tools:
	@command -v $(LATEXMK) >/dev/null 2>&1 || { echo "Error: '$(LATEXMK)' not found. Install latexmk."; exit 1; }
	@command -v $(PDFUNITE) >/dev/null 2>&1 || { echo "Error: '$(PDFUNITE)' not found. Install poppler-utils (pdfunite)."; exit 1; }

clean:
	$(LATEXMK) -c $(TEX_SRCS) || true
	@rm -f $(MERGED_PDF)

veryclean:
	$(LATEXMK) -C $(TEX_SRCS) || true
	@rm -f $(MERGED_PDF)
