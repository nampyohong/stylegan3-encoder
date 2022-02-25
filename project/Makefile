all:
	@echo "Help"
	@echo "  make format"
	@echo "  make install"
	@echo "  make uninstall"
	@echo "  make clean"

format:
	@echo
	find stylegan3_encoder -name "*.py" -exec black -l 120 {} \;

install:
	python setup.py install

uninstall:
	python setup.py uninstall

clean:
	rm -rf build dist stylegan3_encoder.egg-info
