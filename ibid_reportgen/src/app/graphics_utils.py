import matplotlib.pyplot as plt
import re
import os

'''
From Graphics Utils:
'''

FIG_EXTENSION = 'png'


def get_colors(df):
    color_map = {"Correct": "#09ed46", "Incorrect": "#e63b02"}
    return [color_map[str(col)] for col in df.columns]


def section_topics_barplot_gen(section_df, student, section_name, dpath):
    box = section_df.groupby(['topic', 'correct']).size().unstack(fill_value=0)
    box.columns = ['Correct' if col else 'Incorrect' for col in box.columns]

    ax = box.plot(kind='bar', stacked=True, color=get_colors(box), yticks=[])
    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.text(x + width / 2,
                    y + height / 2,
                    f'{int(height)}',
                    horizontalalignment='center',
                    verticalalignment='center')
    plt.legend(prop={'size': 10})

    plt.title(f'{student} - {section_name.title()} Topics')

    x_axis = ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)

    plt.tight_layout()

    fig = ax.get_figure()

    img_fname = re.sub(r'\s', '-', f'{student}-{section_name}-topics-fig.png')
    save_path = os.path.join(dpath, img_fname)
    fig.savefig(save_path)
    plt.close()

    # Return path pointing to image locally
    # NOTE: older versions (pre 6-1-2020) uploaded to AWS. That is now deprecated.
    # Checkout older code if need be.

    return save_path

def section_subtopics_barplot_gen(section_df, student, section_name, dpath):
    topic_groups = section_df.groupby(['topic', 'subtopic', 'correct']).size().unstack(fill_value=0)\
                             .drop('NA', level=0, errors='ignore').drop('NA', level=1, errors='ignore')
    print('Unique topic-level groups (section_subtopics_barplot_gen)')
    print(topic_groups.index.get_level_values(0).unique())
    img_paths = []
    for topic in topic_groups.index.get_level_values(0).unique():
        box = topic_groups.loc[topic]
        box.columns = ['Correct' if col else 'Incorrect' for col in box.columns]
        ax = box.plot(kind='bar', stacked=True, color=get_colors(box), yticks=[])
        xticklabels = ax.get_xticklabels()
        ax.set_xticklabels(xticklabels, rotation=45, ha="right")

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            if height > 0:
                x, y = p.get_xy()
                ax.text(x + width / 2,
                        y + height / 2,
                        f'{int(height)}',
                        horizontalalignment='center',
                        verticalalignment='center')

        plt.legend(prop={'size': 10})
        plt.title(f'{student} - {section_name.title()} Subtopic: {topic.title()}')

        x_axis = ax.axes.get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)

        plt.tight_layout()

        fig = ax.get_figure()
        img_fname = re.sub(r' ', '-', f'{student}-{section_name}-subtopic-{topic}-fig.png')
        save_path = os.path.join(dpath, img_fname)
        fig.savefig(save_path)

        img_paths.append(save_path)

        plt.close()

    return img_paths