COMPILER = g++
FLAGS = -o
EXE = prog
SRC = *.cpp
LINK = -lopencv_core -lopencv_highgui -lOpenCL

#compiling
main:
	$(COMPILER) $(SRC) $(FLAGS) $(EXE) $(LINK)
	
run:
	@echo "running the program"
	./$(EXE)

