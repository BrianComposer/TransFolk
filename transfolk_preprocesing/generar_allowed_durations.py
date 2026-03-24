
def get_allowed_durations(durations=None, dots=None, irregular_groups=None):
    if irregular_groups is None:
        irregular_groups = [[1, 1], [3, 2], [5, 4], [6, 4]]
    if dots is None:
        dots = [0, 1, 2]
    if durations is None:
        durations = [-1, 0, 1, 2, 3, 4, 5] #-2 redonda, -1 blanca, 0 negra, 1 corchea, 2 semicorchea, 3 fusa, 4 semifusa, 5 garrapatea
    allowed_durs = []
    allowed_durs_dic = {}
    for a in durations:
        for b in dots:
            for c in irregular_groups:
                buf = (1 / pow(2, a)) * (2 - 1/pow(2, b)) * c[1] / c[0]
                allowed_durs.append(buf)
                allowed_durs_dic[f"{a},{b},[{c[0]},{c[1]}]"]=buf
    #eliminar duplicados
    allowed_durs = list(dict.fromkeys(allowed_durs))
    #ordenar la lista
    allowed_durs.sort(reverse=True)
    allowed_durs_dic_sorted = dict(sorted(allowed_durs_dic.items(), key=lambda x: x[1]))
    return allowed_durs





if __name__ == "__main__":
    # [0.08333325, 0.1, 0.125, 0.16666665, 0.2, 0.25, 0.333333333, 0.375, 0.5, 0.6666666666666666, 0.75, 1.0, 1.5, 1.75,
    #  2.0, 3.0, 4.0]
    duraciones = [-2, -1, 0, 1, 2, 3, 4] #duraciones
    puntillos = [0,1,2] #puntillos
    grupos = [[1,1], [3,2]]
    allowed_durs = get_allowed_durations(duraciones, puntillos, grupos)
    print(allowed_durs)