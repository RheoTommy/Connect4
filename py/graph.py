import matplotlib.pyplot as plt


def read_file(file):
    res = []
    with open(file, "r") as f:
        data = f.readlines()
        for line in data:
            res.append(float(line.rstrip()))
    return res


if __name__ == '__main__':
    plt.xlabel("Generation")
    plt.ylabel("Score[vs_random]")
    y0 = read_file("../scores/ResNet_vs_random")
    y1 = read_file("../scores/ImprovedResNet_vs_random")
    plt.plot([x + 1 for x in range(len(y0))], y0, label="ResNet")
    plt.plot([x + 1 for x in range(len(y1))], y1, label="ImprovedResNet")
    plt.legend(loc="best")
    plt.savefig("../images/vs_random.png")
    # plt.show()
    plt.close()

    plt.xlabel("Generation")
    plt.ylabel("Score[vs_random_in_random]")
    y0 = read_file("../scores/ResNet_vs_random_in_random")
    y1 = read_file("../scores/ImprovedResNet_vs_random_in_random")
    plt.plot([x + 1 for x in range(len(y0))], y0, label="ResNet")
    plt.plot([x + 1 for x in range(len(y1))], y1, label="ImprovedResNet")
    plt.legend(loc="best")
    plt.savefig("../images/vs_random_in_random.png")
    # plt.show()
    plt.close()

    plt.xlabel("Generation")
    plt.ylabel("Score[vs_pure]")
    y0 = read_file("../scores/ResNet_vs_pure_mct_search")
    y1 = read_file("../scores/ImprovedResNet_vs_pure_mct_search")
    plt.plot([x + 1 for x in range(len(y0))], y0, label="ResNet")
    plt.plot([x + 1 for x in range(len(y1))], y1, label="ImprovedResNet")
    plt.legend(loc="best")
    plt.savefig("../images/vs_pure.png")
    # plt.show()
    plt.close()

    plt.xlabel("Generation")
    plt.ylabel("Score[vs_pure_in_random]")
    y0 = read_file("../scores/ResNet_vs_pure_mct_search_in_random")
    y1 = read_file("../scores/ImprovedResNet_vs_pure_mct_search_in_random")
    plt.plot([x + 1 for x in range(len(y0))], y0, label="ResNet")
    plt.plot([x + 1 for x in range(len(y1))], y1, label="ImprovedResNet")
    plt.legend(loc="best")
    plt.savefig("../images/vs_pure_in_random.png")
    # plt.show()
    plt.close()

    plt.xlabel("Generation")
    plt.ylabel("Score[vs_uct]")
    y0 = read_file("../scores/ResNet_vs_uct_search")
    y1 = read_file("../scores/ImprovedResNet_vs_uct_search")
    plt.plot([x + 1 for x in range(len(y0))], y0, label="ResNet")
    plt.plot([x + 1 for x in range(len(y1))], y1, label="ImprovedResNet")
    plt.legend(loc="best")
    plt.savefig("../images/vs_uct.png")
    # plt.show()
    plt.close()

    plt.xlabel("Generation")
    plt.ylabel("Score[vs_uct_in_random]")
    y0 = read_file("../scores/ResNet_vs_uct_search_in_random")
    y1 = read_file("../scores/ImprovedResNet_vs_uct_search_in_random")
    plt.plot([x + 1 for x in range(len(y0))], y0, label="ResNet")
    plt.plot([x + 1 for x in range(len(y1))], y1, label="ImprovedResNet")
    plt.legend(loc="best")
    plt.savefig("../images/vs_uct_in_random.png")
    # plt.show()
    plt.close()

    plt.xlabel("Generation")
    plt.ylabel("Score[improved_vs_normal")
    y0 = read_file("../scores/improved_vs_normal")
    y1 = read_file("../scores/improved_vs_normal_in_random")
    plt.plot([x + 1 for x in range(len(y0))], y0, label="Connect4")
    plt.plot([x + 1 for x in range(len(y1))], y1, label="Random")
    plt.legend(loc="best")
    plt.savefig("../images/improved_vs_random.png")
    # plt.show()
    plt.close()
