# Makefile for generating numerical analogues of Figs. 4 and 5

PYTHON ?= python

FIG4_LEFT_PNG   := fig4_left.png
FIG4_RIGHT_PNG  := fig4_right.png
FIG5_LEFT_PNG   := fig5_left.png
FIG5_RIGHT_PNG  := fig5_right.png

.PHONY: all fig4_left fig4_right fig5_left fig5_right clean

all: fig4_left fig4_right fig5_left fig5_right

fig4_left: $(FIG4_LEFT_PNG)

$(FIG4_LEFT_PNG): plot_figures.py eigs_methods.py
	$(PYTHON) plot_figures.py --figure 4-left --N 128 --epsilon 0.3 --output $(FIG4_LEFT_PNG) --verbose

fig4_right: $(FIG4_RIGHT_PNG)

$(FIG4_RIGHT_PNG): plot_figures.py eigs_methods.py
	$(PYTHON) plot_figures.py --figure 4-right --epsilon 0.3 --output $(FIG4_RIGHT_PNG) --verbose

fig5_left: $(FIG5_LEFT_PNG)

$(FIG5_LEFT_PNG): plot_figures.py eigs_methods.py
	$(PYTHON) plot_figures.py --figure 5-left --N 1024 --epsilon 0.1 --output $(FIG5_LEFT_PNG) --verbose

fig5_right: $(FIG5_RIGHT_PNG)

$(FIG5_RIGHT_PNG): plot_figures.py eigs_methods.py
	$(PYTHON) plot_figures.py --figure 5-right --N 512 --output $(FIG5_RIGHT_PNG) --verbose

clean:
	del /Q $(FIG4_LEFT_PNG) $(FIG4_RIGHT_PNG) $(FIG5_LEFT_PNG) $(FIG5_RIGHT_PNG) 2>nul || true
