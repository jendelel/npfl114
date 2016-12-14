def mc(seq):
    return max(seq, key=seq.count)

pasted_file = "pasted.txt"

votes = {}
eids = []
with open(pasted_file, "r") as f:
    for line in f:
        splitted = line.split("\t")
        eid = splitted[0]
        vote = splitted[1].strip()
        eid = int(eid)
        if eid not in votes:
            votes[eid] = [vote]
            eids.append(eid)
        else:
            votes[eid].append(vote)

results = [mc(votes[eid]) for eid in eids]
with open("labels.txt", "w") as f:
    for r in results:
        f.write(r + "\n")



