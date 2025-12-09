PYTHON = python3

# Default target: build everything
all: fig2 fig3 extra

fig2:
	$(PYTHON) interpolation_project.py fig2

fig3:
	$(PYTHON) interpolation_project.py fig3

extra:
	$(PYTHON) interpolation_project.py extra

clean:
	rm -f fig2_f1_interp.png fig2_f2_interp.png fig2_max_error.png
	rm -f fig3_interp.png fig3_error.png
	rm -f extra_g1_max_error.png extra_g2_max_error.png
