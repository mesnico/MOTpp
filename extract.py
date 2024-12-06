import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="extract", version_base="1.3")
def extract(cfg: DictConfig):
    run_dir = cfg.run_dir
    ckpt = cfg.ckpt

    from src.load import extract_ckpt

    logger.info("Extracting the checkpoint...")
    try:
        extract_ckpt(run_dir, ckpt_name=ckpt)
    except Exception as e:
        logger.error(e)
        exit(1)
    logger.info("Done")


if __name__ == "__main__":
    extract()
