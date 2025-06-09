from hmmlearn.hmm import GaussianHMM
import matplotlib.cm as cm


def plot_hidden_states(df_emb_hmm, hmm_model):
    n_states = df_emb_hmm["HMM_state"].nunique()
    colors = [cm.get_cmap('Dark2')(i) for i in range(n_states)]
    plt.figure(figsize=(12, 4))
    for i in range(hmm_model.n_components):
        mask = df_emb_hmm["HMM_state"] == i
        plt.plot(df_emb_hmm.index[mask], df_emb_hmm["log_returns"][mask], '.', label=f"State {i}", alpha=0.6, markersize=4, color=colors[i])
    plt.title("Log-Returns identified by Hidden Markov States")
    plt.xlabel("Date")
    plt.ylabel("Log Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def hidden_mm(df_emb, X, n_states=2, covariance_type="full", max_iter=1000, myseed=42, params='stmc', init_params='stmc'):
    hmm_model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=max_iter, random_state=myseed)
    hmm_model.fit(X)
    hidden_states = hmm_model.predict(X)
    df_emb_hmm = df_emb.copy().iloc[-len(hidden_states):]
    df_emb_hmm["HMM_state"] = hidden_states
    means = hmm_model.means_.flatten()
    variances = [np.diag(cov)[0] for cov in hmm_model.covars_]
    transition_matrix = hmm_model.transmat_
    for i in range(n_states):
        print(f"Regime {i}: mean = {means[i]:.5f}, variance = {variances[i]:.5f}")
    print("\nTransition matrix:")
    print(transition_matrix)
    plot_hidden_states(df_emb_hmm, hmm_model)
    return df_emb_hmm, hmm_model
