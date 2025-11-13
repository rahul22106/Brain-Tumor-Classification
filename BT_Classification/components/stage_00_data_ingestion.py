import os
import sys
import zipfile
import shutil
from pathlib import Path
import urllib.request as request
from BT_Classification import logger
from BT_Classification.utils.common import get_size
from BT_Classification.entity import DataIngestionConfig


class DataIngestion:
    """
    Component responsible for data ingestion
    - Downloads data from source (if URL provided) or copies from local path
    - Extracts zip files
    - Organizes data into train/test directories
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize Data Ingestion component
        
        Args:
            config (DataIngestionConfig): Configuration for data ingestion
        """
        self.config = config
    
    
    def download_file(self):
        """
        Download file from Google Drive URL or copy from local path if not already exists
        """
        try:
            if not os.path.exists(self.config.local_data_file):
                logger.info("File not found. Starting download/copy...")
                
                # Check if source is Google Drive URL
                if 'drive.google.com' in self.config.source_URL:
                    logger.info(f"Downloading from Google Drive: {self.config.source_URL}")
                    
                    # Extract file ID from Google Drive URL
                    file_id = self.config.source_URL.split("/")[-2]
                    logger.info(f"Extracted file ID: {file_id}")
                    
                    # Create download URL
                    prefix = 'https://drive.google.com/uc?/export=download&id='
                    download_url = prefix + file_id
                    
                    # Download using gdown
                    import gdown
                    gdown.download(download_url, str(self.config.local_data_file), quiet=False)
                    
                    logger.info(f"Download completed!")
                    logger.info(f"File saved at: {self.config.local_data_file}")
                    logger.info(f"File size: {get_size(Path(self.config.local_data_file))}")
                
                # Check if source is regular URL
                elif self.config.source_URL.startswith('http'):
                    logger.info(f"Downloading from URL: {self.config.source_URL}")
                    filename, headers = request.urlretrieve(
                        url=self.config.source_URL,
                        filename=str(self.config.local_data_file)
                    )
                    logger.info(f"Downloaded to: {filename}")
                    logger.info(f"File size: {get_size(Path(self.config.local_data_file))}")
                
                # Copy from local path
                else:
                    logger.info(f"Copying data from local path: {self.config.source_URL}")
                    if os.path.isfile(self.config.source_URL):
                        # If it's a zip file
                        shutil.copy(self.config.source_URL, str(self.config.local_data_file))
                        logger.info(f"File copied successfully")
                    elif os.path.isdir(self.config.source_URL):
                        # If it's a directory
                        shutil.copytree(self.config.source_URL, str(self.config.unzip_dir), dirs_exist_ok=True)
                        logger.info(f"Directory copied successfully")
            else:
                logger.info(f"File already exists: {self.config.local_data_file}")
                logger.info(f"File size: {get_size(Path(self.config.local_data_file))}")
                
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def extract_zip_file(self):
        """
        Extract zip file into the data directory
        """
        try:
            unzip_path = str(self.config.unzip_dir)
            os.makedirs(unzip_path, exist_ok=True)
            
            local_file = str(self.config.local_data_file)
            
            # Check if local_data_file exists and is a zip
            if os.path.exists(local_file) and local_file.endswith('.zip'):
                logger.info(f"Extracting zip file: {local_file}")
                
                with zipfile.ZipFile(local_file, 'r') as zip_ref:
                    zip_ref.extractall(unzip_path)
                
                logger.info(f"Extraction completed at: {unzip_path}")
            else:
                logger.info("No zip file to extract or data already extracted.")
                
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def validate_data_structure(self):
        """
        Validate that train and test directories exist with proper structure
        Expected structure:
        artifacts/data_ingestion/
        ├── train/
        │   ├── class1/
        │   ├── class2/
        │   └── ...
        └── test/
            ├── class1/
            ├── class2/
            └── ...
        """
        try:
            logger.info("Validating data structure...")
            
            # Check if train and test directories exist
            train_exists = os.path.exists(str(self.config.train_data_dir))
            test_exists = os.path.exists(str(self.config.test_data_dir))
            
            if train_exists and test_exists:
                # Count images
                train_count = 0
                test_count = 0
                
                # Count training images
                for root, dirs, files in os.walk(str(self.config.train_data_dir)):
                    train_count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                # Count test images
                for root, dirs, files in os.walk(str(self.config.test_data_dir)):
                    test_count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                logger.info(f"✓ Training images found: {train_count}")
                logger.info(f"✓ Test images found: {test_count}")
                
                # Get class names from train directory
                train_classes = [d for d in os.listdir(str(self.config.train_data_dir)) 
                               if os.path.isdir(os.path.join(str(self.config.train_data_dir), d))]
                test_classes = [d for d in os.listdir(str(self.config.test_data_dir)) 
                              if os.path.isdir(os.path.join(str(self.config.test_data_dir), d))]
                
                logger.info(f"✓ Training classes: {train_classes}")
                logger.info(f"✓ Test classes: {test_classes}")
                
                # Verify same classes in both
                if set(train_classes) == set(test_classes):
                    logger.info("✓ Train and test have same classes")
                else:
                    logger.warning("⚠ Train and test classes don't match!")
                    logger.warning(f"Train only: {set(train_classes) - set(test_classes)}")
                    logger.warning(f"Test only: {set(test_classes) - set(train_classes)}")
                
                return True
            else:
                logger.error("✗ Train or test directory not found!")
                if not train_exists:
                    logger.error(f"Missing: {self.config.train_data_dir}")
                if not test_exists:
                    logger.error(f"Missing: {self.config.test_data_dir}")
                return False
                
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def organize_data_if_needed(self):
        """
        Organize data if it's not in the expected train/test structure
        This handles cases where extracted data might be in a different structure
        """
        try:
            logger.info("Checking if data organization is needed...")
            
            # If train and test already exist in correct location, skip
            if os.path.exists(str(self.config.train_data_dir)) and os.path.exists(str(self.config.test_data_dir)):
                logger.info("✓ Data is already organized in train/test structure")
                return
            
            # Search for train/test folders in extracted directory
            for root, dirs, files in os.walk(str(self.config.unzip_dir)):
                if 'train' in dirs and 'test' in dirs:
                    # Found train/test structure
                    found_train = os.path.join(root, 'train')
                    found_test = os.path.join(root, 'test')
                    
                    logger.info(f"Found train directory at: {found_train}")
                    logger.info(f"Found test directory at: {found_test}")
                    
                    # Move to expected location if different
                    if found_train != str(self.config.train_data_dir):
                        logger.info(f"Moving train data to: {self.config.train_data_dir}")
                        shutil.move(found_train, str(self.config.train_data_dir))
                    
                    if found_test != str(self.config.test_data_dir):
                        logger.info(f"Moving test data to: {self.config.test_data_dir}")
                        shutil.move(found_test, str(self.config.test_data_dir))
                    
                    logger.info("✓ Data organization completed")
                    return
            
            logger.warning("⚠ Could not find train/test structure in extracted data")
            logger.info("Please manually organize data into train/test directories")
            
        except Exception as e:
            logger.exception(e)
            raise e
    
    
    def initiate_data_ingestion(self):
        """
        Main method to execute all data ingestion steps
        """
        try:
            logger.info("=" * 70)
            logger.info("STARTING DATA INGESTION")
            logger.info("=" * 70)
            
            # Step 1: Download or copy data
            logger.info("\n>>> Step 1: Download/Copy Data")
            self.download_file()
            
            # Step 2: Extract zip file (if applicable)
            logger.info("\n>>> Step 2: Extract Data")
            self.extract_zip_file()
            
            # Step 3: Organize data structure
            logger.info("\n>>> Step 3: Organize Data")
            self.organize_data_if_needed()
            
            # Step 4: Validate data structure
            logger.info("\n>>> Step 4: Validate Data Structure")
            is_valid = self.validate_data_structure()
            
            if is_valid:
                logger.info("\n" + "=" * 70)
                logger.info("✓ DATA INGESTION COMPLETED SUCCESSFULLY")
                logger.info("=" * 70)
            else:
                logger.error("\n" + "=" * 70)
                logger.error("✗ DATA INGESTION FAILED - Please check data structure")
                logger.error("=" * 70)
            
            return self.config.train_data_dir, self.config.test_data_dir
            
        except Exception as e:
            logger.exception(e)
            raise e