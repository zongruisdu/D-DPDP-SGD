def get_graph(word_size, topology):
    graph = dict()
    if(topology == 'ring'):
        for i in range(word_size):
            graph[i] = set()
            #graph[i].add(i)
            graph[i].add((i+1)%word_size)
            graph[i].add((i-1+word_size)%word_size)
        return graph
    else:
        return graph