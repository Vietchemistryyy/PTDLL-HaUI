"""Utilities cho Spark và các functions chung"""
from pyspark.sql import SparkSession
from pyspark import SparkConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkManager:
    """Quản lý Spark Session"""
    _instance = None

    @classmethod
    def get_spark(cls, app_name="CardioAnalysis", master="local[*]",
                  driver_memory="4g", executor_memory="4g"):
        """Khởi tạo hoặc lấy Spark Session"""
        if cls._instance is None:
            logger.info(f"Khởi tạo Spark Session: {app_name}")

            conf = SparkConf()
            conf.set("spark.driver.memory", driver_memory)
            conf.set("spark.executor.memory", executor_memory)
            conf.set("spark.sql.adaptive.enabled", "true")

            cls._instance = SparkSession.builder \
                .appName(app_name) \
                .master(master) \
                .config(conf=conf) \
                .getOrCreate()

            cls._instance.sparkContext.setLogLevel("WARN")
            logger.info(f"✓ Spark {cls._instance.version} đã sẵn sàng")

        return cls._instance

    @classmethod
    def stop(cls):
        """Dừng Spark Session"""
        if cls._instance:
            cls._instance.stop()
            cls._instance = None
            logger.info("Spark Session đã dừng")


def init_spark(config):
    """Khởi tạo Spark với config"""
    return SparkManager.get_spark(
        app_name=config["app_name"],
        master=config["master"],
        driver_memory=config["driver_memory"],
        executor_memory=config["executor_memory"]
    )