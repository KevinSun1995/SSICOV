import matplotlib.pyplot as plt


def cable_force(m, l, deta_f):
    return 4 * m * (l ** 2) * (deta_f ** 2) / 1000


if __name__ == '__main__':
    with open('cable_force_result.txt') as f:
        data = f.readline().strip().split()
        m = 123
        l = 124
        data = [cable_force(m, l, float(x)) for x in data]
        plt.plot(range(len(data)), data)
        plt.show()
