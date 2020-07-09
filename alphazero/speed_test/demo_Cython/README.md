Here, we test how Cython improves on Python for detecting if a move at Connect4 is winning.

- testc4.py contains the python function and some test code
- c4speed.pyx contains the same function, plus additional type information (that will be used by Cython to make things more efficient)
- the setup.py file that contains the instruction to compile the pyx file

Step 1: compile cython code:
python setup.py build_ext --inplace

Step 2: run testc4.py po compare Cython and Python. The function compiled with Cython is imported via: import c4speed as c4
