if __name__ == "__main__":

    # Parse arguments
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--input_file", default="nli-dataset/nli-train.txt", type=str, help="Input file.")
    args.add_argument("--out_file", default="nli-dataset/nli-train3.txt", type=str, help="Output file.")

    args = args.parse_args()

    # Load the sentences
    with open(args.input_file, "r") as file:
        with open(args.out_file, "w") as out:
            essay_id = 0
            for line in file:
                line = line.rstrip("\r\n")
                language, prompt, level, words = line.split("\t", 3)
                sentences = words.split(". .")
                for sentence in sentences:
                    if sentence.strip() == "":
                        continue
                    sentence = sentence.lstrip("\t").rstrip("\t")
                    out_line = str(essay_id) + "\t" + language + "\t" + prompt + "\t" + level + "\t" + sentence + "\t. .\r\n"
                    out.write(out_line)
                essay_id += 1
