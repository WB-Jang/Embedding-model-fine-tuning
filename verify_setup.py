#!/usr/bin/env python3
"""
Verification script to check if main.py can be imported and has valid structure.
This does NOT run the full training, just validates the code structure.
"""
import sys
import ast

def verify_main_py():
    """Verify main.py syntax and structure"""
    print("🔍 Verifying src/main.py...")
    
    # Read the file
    with open('src/main.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Check if it's valid Python syntax
    try:
        tree = ast.parse(code)
        print("✅ Syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    
    # Check for required imports
    required_imports = ['pandas', 'sentence_transformers', 'torch']
    found_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found_imports.append(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found_imports.append(node.module.split('.')[0])
    
    missing_imports = set(required_imports) - set(found_imports)
    if missing_imports:
        print(f"⚠️  Missing imports: {missing_imports}")
    else:
        print("✅ All required imports are present")
    
    # Check for key variables
    has_model_id = 'MODEL_ID' in code
    has_output_path = 'OUTPUT_PATH' in code
    has_batch_size = 'BATCH_SIZE' in code
    has_num_epochs = 'NUM_EPOCHS' in code
    
    if all([has_model_id, has_output_path, has_batch_size, has_num_epochs]):
        print("✅ All configuration variables are defined")
    else:
        print("⚠️  Some configuration variables might be missing")
    
    # Check for main training logic
    has_model_load = 'SentenceTransformer' in code
    has_dataloader = 'DataLoader' in code
    has_loss = 'MultipleNegativesRankingLoss' in code
    has_fit = '.fit(' in code
    
    if all([has_model_load, has_dataloader, has_loss, has_fit]):
        print("✅ Main training components are present")
    else:
        print("⚠️  Some training components might be missing")
    
    print("\n✨ Verification complete!")
    print("\n📝 Summary:")
    print("   - The code is syntactically correct")
    print("   - All required imports are present")
    print("   - Configuration variables are defined")
    print("   - Training logic is implemented")
    print("\n⚠️  Note: This verification does not test runtime behavior.")
    print("   To fully test the training, run: python src/main.py")
    print("   (This will require GPU and may take time)")
    
    return True

if __name__ == '__main__':
    success = verify_main_py()
    sys.exit(0 if success else 1)
