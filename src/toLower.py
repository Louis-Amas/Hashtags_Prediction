from sys import argv
if __name__ == '__main__':
    """
        Transform file to lowercase
    """
    if len(argv) < 3:
        exit(1)

    in_path = argv[1]
    out_path = argv[2]

    with open(in_path, 'r') as f_in:
        data = f_in.read().lower()

    with open(out_path, 'w') as f_out:
        f_out.write(data)
