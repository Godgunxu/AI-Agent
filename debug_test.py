#!/usr/bin/env python3
"""
Quick test to debug the QA system
"""

import sys
sys.path.append('/Users/xujing/Downloads/LavieFinancial/AI Agent')

from app_fast import create_general_qa_chain

print("🧪 Testing General QA Chain...")

try:
    # Test creating the chain
    chain = create_general_qa_chain("llama3.2:latest", fast_mode=False)
    
    if chain is None:
        print("❌ Failed to create QA chain!")
    else:
        print("✅ QA chain created successfully")
        
        # Test a simple question
        test_question = "What is bubble sort?"
        print(f"\n🤔 Testing question: {test_question}")
        
        result = chain.run(test_question)
        print(f"\n✅ Response received:")
        print(f"Length: {len(result)} characters")
        print(f"Content preview: {result[:200]}...")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
