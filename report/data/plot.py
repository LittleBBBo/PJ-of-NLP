import os
import matplotlib.pyplot as plt
import pandas as pd


def subM_plot_df(perplexity_df, name="tmp"):
    fig, ax = plt.subplots()
    perplexity_df.columns = [int(x) for x in perplexity_df.columns]
    selected = perplexity_df.ix[:, perplexity_df.columns <= 20].T
    selected.plot.line(style=['D-', 'o-', '^-', 'v-', '<-'], ax=ax, xlim=2)
    for line in ax.lines:
        line.set_linewidth(2)
        line.set_markersize(10)
    ax.set_xlabel("subK")
    ax.set_ylabel("perplexity")
    fig.savefig("%s" % name)
    return fig


def subK_plot_df(perplexity_df, name="tmp"):
    fig, ax = plt.subplots()
    perplexity_df.columns = [int(x) for x in perplexity_df.columns]
    perplexity_df.T.plot.line(style=['D-', 'o-', '^-', 'v-', '<-'], ax=ax,
                              xlim=200)
    for line in ax.lines:
        line.set_linewidth(2)
        line.set_markersize(10)
    ax.set_xlabel("subM")
    ax.set_ylabel("perplexity")
    fig.savefig("%s" % name)
    return fig


def effi_subM_plot(effi_df, name="tmp"):
    fig, ax = plt.subplots()
    effi_df.columns = [int(x) for x in effi_df.columns]
    selected = effi_df.iloc[0:2, :]
    selected.T.plot.line(style=['D-', 'o-'], ax=ax, xlim=2)
    for line in ax.lines:
        line.set_linewidth(2)
        line.set_markersize(10)
    ax.set_xlabel("subK")
    ax.set_ylabel("time(s)")
    fig.savefig("%s" % name)
    return fig


def effi_x_subM_plot(effi_df, name="tmp"):
    fig, ax = plt.subplots()
    effi_df.columns = [int(x) for x in effi_df.columns]
    effi_df.iloc[2].T.plot.line(style=['D-'], ax=ax, xlim=2)
    for line in ax.lines:
        line.set_linewidth(2)
        line.set_markersize(10)
    ax.set_xlabel("subK")
    ax.set_ylabel("speed up")
    fig.savefig("%s" % name)
    return fig


def effi_subK_plot(effi_df, name="tmp"):
    fig, ax = plt.subplots()
    effi_df.columns = [int(x) for x in effi_df.columns]
    effi_df.T.plot.line(style=['D-', 'o-'], ax=ax, xlim=200)
    for line in ax.lines:
        line.set_linewidth(2)
        line.set_markersize(10)
    ax.set_xlabel("subM")
    ax.set_ylabel("time(s)")
    fig.savefig("%s" % name)
    return fig


def effi_x_subK_plot(effi_df, name="tmp"):
    fig, ax = plt.subplots()
    effi_df.columns = [int(x) for x in effi_df.columns]
    effi_df.iloc[2].T.plot.line(style=['D-'], ax=ax, xlim=200)
    for line in ax.lines:
        line.set_linewidth(2)
        line.set_markersize(10)
    ax.set_xlabel("subM")
    ax.set_ylabel("speed up")
    fig.savefig("%s" % name)
    return fig


if __name__ == "__main__":
    csvdir = "csv"
    figdir = "fig"

    # perplexity
    index_name = ["vanilla LDA", "New method", "baseline"]

    fname = "perplexity_wiki_subK7"
    df = pd.read_csv(os.path.join(csvdir, "%s.csv" % fname), index_col=0)
    df.index = index_name
    subK_plot_df(df, os.path.join(figdir, "%s.pdf" % fname))

    fname = "perplexity_wiki_subM1000"
    df = pd.read_csv(os.path.join(csvdir, "%s.csv" % fname), index_col=0)
    df.index = index_name
    subM_plot_df(df, os.path.join(figdir, "%s.pdf" % fname))

    fname = "perplexity_enron_subK7"
    df = pd.read_csv(os.path.join(csvdir, "%s.csv" % fname), index_col=0)
    df.index = index_name
    subK_plot_df(df, os.path.join(figdir, "%s.pdf" % fname))

    fname = "perplexity_enron_subM1000"
    df = pd.read_csv(os.path.join(csvdir, "%s.csv" % fname), index_col=0)
    df.index = index_name
    subM_plot_df(df, os.path.join(figdir, "%s.pdf" % fname))

    # efficiency
    fname = "effi_subK7"
    df = pd.read_csv(os.path.join(csvdir, "%s.csv" % fname), index_col=0)
    tmp_index = index_name[:2]
    tmp_index.append(df.index[2])
    df.index = tmp_index
    effi_subK_plot(df.iloc[0:2], os.path.join(figdir, "%s.pdf" % fname))
    effi_x_subK_plot(df, os.path.join(figdir, "%s_x.pdf" % fname))

    fname = "effi_subM2000"
    df = pd.read_csv(os.path.join(csvdir, "%s.csv" % fname), index_col=0)
    tmp_index = index_name[:2]
    tmp_index.append(df.index[2])
    df.index = tmp_index
    effi_subM_plot(df.iloc[0:2], os.path.join(figdir, "%s.pdf" % fname))
    effi_x_subM_plot(df, os.path.join(figdir, "%s_x.pdf" % fname))
