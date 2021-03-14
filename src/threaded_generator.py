# A simple generator wrapper, not sure if it's good for anything at all.
# With basic python threading
from threading import Thread
import threading, time

try:
    from queue import Queue

except ImportError:
    from Queue import Queue
 
# ... or use multiprocessing versions
# WARNING: use sentinel based on value, not identity
from multiprocessing import Process, Queue as MpQueue

class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, iterator,
                 sentinel=object(),
                 queue_maxsize=0,
                 daemon=True,
                 Thread=Thread,
                 Queue=Queue):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon = daemon

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                self._queue.put(value)

        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value

        self._thread.join()


def threadOn(iterator):
    return ThreadedGenerator(iterator)

def threadsOn(iterators):
    for iterator in iterators:
        yield ThreadedGenerator(iterator)

def task1(s):
    for i in range(2):
        time.sleep(3)
        yield s
    
def task2(s):
    for i in s:
        time.sleep(2)
        yield i

def task3(s):
    for i in s:
        for ii in range(2):
            time.sleep(1)
            yield i

def task33(s):
    for i in range(3):
        time.sleep(2)
        yield i

def loop(s):
    for i in s:
        print(i)

def loops(s):
    for i in s:
        for j in i:
            print(j)



if __name__ == "__main__":
    start = time.time()
    s = task1(1)
    #s = threadOn(s)
    s = threadOn(s)
    #s = map(threadOn, s)
    #s = list(s)
    
    s = map(task1, s)
    
    #s = threadsOn(s)
    #s = task2(s)
    #s = threadOn(s)
    #s = threadOn(s)
    #s = list(s)
    #s = task3(s)
    #s = threadsOn(s)
    
    loops(s)
    end = time.time()

    print(end - start)