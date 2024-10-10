clean:
	rm -rf __pycache__ performance.png

data:
	./script/mv_data

play:
	./main.py --play-game True --save-dir played_data

run:
	./main.py
