PYTHON ?= python3
LATEXMK ?= latexmk
PLANTUML ?= plantuml
RSVG ?= rsvg-convert

BUILD := build
FIGURES := $(patsubst figures/%.puml,$(BUILD)/figures/%.png,$(wildcard figures/*.puml))
PDF := ../outputs/rapport/TuniScope_Rapport.pdf

.PHONY: all tables
all: $(PDF)

tables:
	$(PYTHON) generate_tables.py

$(BUILD)/figures/%.png: figures/%.puml
	mkdir -p $(BUILD)/figures
	$(PLANTUML) -charset UTF-8 -failfast2 -tsvg -o "$(CURDIR)/$(BUILD)/figures" "$<"
	$(RSVG) -f png -z 4 -o "$@" "$(BUILD)/figures/$*.svg"

$(PDF): tables $(FIGURES)
	mkdir -p $(BUILD)/latex $(BUILD)/tex-cache "$(@D)"
	TEXMFCACHE="$(CURDIR)/$(BUILD)/tex-cache" TEXMFVAR="$(CURDIR)/$(BUILD)/tex-cache" \
	  $(LATEXMK) -lualatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD)/latex main.tex
	cp "$(BUILD)/latex/main.pdf" "$@"
