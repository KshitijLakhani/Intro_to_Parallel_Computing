#This is the format of these syntaxes
#
#target:<space>dependencies
#<tab>command

#Variable declaration
CC=gcc
CFLAGS=-c -Wall

all: hello

hello: main.o function1.o function2.o
	gcc -o factorial main.o function1.o function2.o

main.o: main.c
	$(CC) $(CFLAGS) main.c

function1.o: function1.c
	$(CC) $(CFLAGS) function1.c

function2.o: function2.c
	$(CC) $(CFLAGS) function2.c

clean: 
	rm -rf *o factorial
#In the terminal the basic way to run the makefile is 
#make target i.e. make all OR make clean and so on
#The latter will delete all the object files created(i.e. perform a clean)



#The -c flag signifies only compile and not linking 
#That is why if you run the command
#gcc -o factorial main.c function1.c function2.c 
#You will get the same executable factorial like 
#running this makefile would give you using 
#make all or make
#Without the -c, it link the files but with the -c
# it does not
#
# Also, let's say we make some changes in main.c and not in the other
# 2 '.c' files then when you run make again it will only re-compile
# the file that has been modifies and leave the others as it is
