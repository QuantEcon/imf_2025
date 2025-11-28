# Makefile for faster LaTeX compilation
#
# Usage:
#   make          - Full compile (slow, for final version)
#   make draft    - Draft mode (fast, no graphics)
#   make quick    - Quick compile using latexmk
#   make clean    - Remove auxiliary files

MAIN = main
TEX = xelatex
FLAGS = -interaction=nonstopmode -shell-escape

# Full compile
all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(TEX) $(FLAGS) $(MAIN).tex
	$(TEX) $(FLAGS) $(MAIN).tex

# Draft mode - fastest for previewing
draft:
	$(TEX) $(FLAGS) -jobname=$(MAIN) "\PassOptionsToClass{draft}{beamer}\input{$(MAIN)}"

# Using latexmk for smart recompilation (only recompiles what changed)
quick:
	latexmk -xelatex -shell-escape -interaction=nonstopmode $(MAIN).tex

# Continuous preview mode (recompiles on file changes)
watch:
	latexmk -xelatex -shell-escape -interaction=nonstopmode -pvc $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.nav *.snm *.toc *.vrb *.bbl *.blg
	rm -rf _minted/

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all draft quick watch clean distclean
