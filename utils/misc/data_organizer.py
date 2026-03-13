import os
import shutil

def organize_data_folders(root_path):
    print(f"📊 Starting Data Categorization in: {root_path}")
    
    target_extensions = {
        '.npy': 'npy',
        '.nwb': 'nwb',
        '.pd': 'pd',
        '.mat': 'mat',
        '.txt': 'txt'
    }
    
    exempt_dirs = {'.git', 'hnxj-gemini', '__pycache__', 'venv', '.gemini'}

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip exempt directories
        if any(exempt in dirpath.split(os.sep) for exempt in exempt_dirs):
            continue
            
        # Group files by extension
        files_to_move = {}
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in target_extensions:
                cat_folder = target_extensions[ext]
                if cat_folder not in files_to_move:
                    files_to_move[cat_folder] = []
                files_to_move[cat_folder].append(filename)
        
        # If we have files to move, create the 'data' structure
        if files_to_move:
            data_root = os.path.join(dirpath, "data")
            
            for cat_folder, files in files_to_move.items():
                target_dir = os.path.join(data_root, cat_folder)
                
                # Create directory only if not already there
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                    print(f"  📁 Created: {target_dir}")
                
                for f in files:
                    src = os.path.join(dirpath, f)
                    dst = os.path.join(target_dir, f)
                    try:
                        shutil.move(src, dst)
                    except Exception as e:
                        print(f"  ⚠️ Error moving {f}: {e}")

    print("✨ Data organization complete.")

if __name__ == "__main__":
    workspace_root = "/Users/hamednejat/workspace"
    organize_data_folders(workspace_root)
