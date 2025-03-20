#include <random>
#include <string>

#include "spdlog/spdlog.h"
#include "Eigen/Eigen"

/**
 * @brief Action-value Updater
 *
 * @param Q_n Vector of action-values for A actions
 * @param a Which action was selected prior to reward R_n
 * @param R_n Reward value received for taking action a
 * @param alpha Step-size parameter (1/n)
 */
template <int A>
Eigen::Vector<float, A> Q_nplus1(Eigen::Vector<float, A> Q_n, int a, float R_n, float alpha) {
	Eigen::Vector<float, A> Q_nplus1_res = Q_n;
	Q_nplus1_res[a] = Q_n[a] + alpha*(R_n - Q_n[a]);
	return Q_nplus1_res;
}

template<int A>
struct fmt::formatter<Eigen::Vector<float, A>> : fmt::formatter<std::string> {
	auto format(Eigen::Vector<float,A>  my, format_context &ctx) const -> decltype(ctx.out()) {
		std::string outstring = "";
		
		outstring += "{ ";
		for(auto it: my) {
			outstring += std::to_string(it);
			outstring += " ";
		}
		outstring += " }";
		return fmt::format_to(ctx.out(), "{}", outstring);
	}
};

int main(void) {
	const unsigned int num_acts = 5;
	const float max_reward = 20.f;
	const float min_reward = -20.f;
	const float alpha = 0.1;

	Eigen::Vector<float, num_acts> Qvals;
	Qvals.setZero();

	/* Set log level */
	spdlog::set_level(spdlog::level::debug);

	for(int x = 0; x < 100; x++) {
		/* Take random action */
		int sel = (std::rand() % num_acts);
		spdlog::debug("Selecting action {}", sel);

		/* Give random reward */
		float rew = (std::rand() % (int)(max_reward - min_reward)) - (max_reward - min_reward)/2.f;
		spdlog::debug("Giving reward {}", rew);

		/* Update Q-values */
		Qvals = Q_nplus1(Qvals, sel, rew, alpha);

		/* Display updates values */
		spdlog::info("Update Q-values: {}", Qvals);
	}
}
