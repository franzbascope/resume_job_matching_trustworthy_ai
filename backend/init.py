import kagglehub
import shutil


# Download latest version
source_path = kagglehub.dataset_download("ravindrasinghrana/job-description-dataset")

print(f"source_path: {source_path}")

destination_path = "../backend"  # Replace with the target location

shutil.move(source_path, destination_path)
print(f"File moved to: {destination_path}")
