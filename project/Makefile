INCLUDE_DIRS = 
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS =  -g -Wall  $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= quadtree.cpp

SRCS= ${HFILES} ${CFILES}

all: quadtree 

clean:
	-rm -f *.o *.d
	-rm -f quadtree

quadtree: quadtree.cpp
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.cpp  -pthread