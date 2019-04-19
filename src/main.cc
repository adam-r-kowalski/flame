#include <boost/filesystem.hpp>
#include <flame.hh>
#include <opencv2/opencv.hpp>

namespace fs = boost::filesystem;

auto train_environment() -> void;

auto main() -> int {
  const auto data_path = fs::path{getenv("HOME")} / "Downloads";
  const auto images_path = data_path / "train-volume.tif";
  const auto labels_path = data_path / "train-labels.tif";

  auto images = std::vector<cv::Mat>{};
  cv::imreadmulti(images_path.string(), images);

  auto labels = std::vector<cv::Mat>{};
  cv::imreadmulti(labels_path.string(), labels);

  auto tensor = flame::convert<torch::Tensor>(images[0]);
  auto image = flame::convert<cv::Mat>(tensor);

  cv::imshow("image", image);
  cv::imshow("label", labels[0]);
  cv::waitKey();

  return 0;
}

auto train_environment() -> void {
  const auto interpreter = flame::python_interpreter();
  const auto gym = flame::gym::Gym{interpreter};
  const auto tensorboard = flame::Tensorboard{interpreter};

  auto environment = gym.make("CartPole-v0");

  auto agent = flame::agent::PolicyGradient{
      {.observation_space =
           environment.observation_space().shape[0].item<int>(),
       .action_space = environment.action_space().n,
       .gamma = 0.9}};

  flame::Simulation{}.episodes(3).render(true).run(environment, agent);

  for (auto i = 0; i < 3; ++i) {
    flame::Simulation{}
        .episodes(100)
        .on_episode_end({flame::callback::TensorboardLogger{tensorboard}})
        .run(environment, agent);

    flame::Simulation{}.episodes(3).render(true).run(environment, agent);
  }
}
