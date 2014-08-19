% ID2INDEX  Transforms part id to a usable index (deprecated)

function index = id2index(id)
index = mod(id,10^5);