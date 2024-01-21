from collections import defaultdict

import matplotlib.pyplot as plt


def parse_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    data = defaultdict(list)

    for line in lines:
        if '|' in line:
            split_line = line.split('|')
            key = split_line[1].strip()
            try:
                value = float(split_line[2].strip())
                data[key].append(value)
            except ValueError:
                pass

    return data


def plot_data(data, file_name):
    timesteps = data['total_timesteps']
    for key, values in data.items():
        if key != 'total_timesteps':
            plt.figure(figsize=(10, 5))
            try:
                plt.plot(timesteps, values)
                plt.xlabel('total_timesteps')
            except ValueError:
                plt.plot(values)
            plt.title(f"{key} from {file_name}")
            plt.show()


if __name__ == '__main__':
    data = parse_file('PPO uczenie.txt')
    plot_data(data, 'PPO uczenie.txt')

    data = parse_file('SAC uczenie.txt')
    plot_data(data, 'SAC uczenie.txt')
