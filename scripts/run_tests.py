#!/usr/bin/env python3
"""
Script para ejecutar tests del proyecto.
"""
import sys
import subprocess
from pathlib import Path

def run_tests():
    """
    Ejecutar todos los tests del proyecto.
    """
    print("=== EJECUTANDO TESTS DEL PROYECTO ===")
    
    # Ejecutar tests unitarios
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print("Resultado de tests:")
        print(result.stdout)
        
        if result.stderr:
            print("Errores:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error ejecutando tests: {e}")
        return False

def run_linting():
    """
    Ejecutar linting del código.
    """
    print("=== EJECUTANDO LINTING ===")
    
    try:
        # Flake8
        result = subprocess.run([
            "flake8", "src/", "scripts/", "--max-line-length=100"
        ], capture_output=True, text=True)
        
        if result.stdout:
            print("Problemas de estilo encontrados:")
            print(result.stdout)
        else:
            print("Código sin problemas de estilo")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error ejecutando linting: {e}")
        return False

def main():
    """
    Función principal.
    """
    print("Verificando calidad del código...")
    
    # Ejecutar tests
    tests_ok = run_tests()
    
    # Ejecutar linting
    linting_ok = run_linting()
    
    if tests_ok and linting_ok:
        print("\n Todos los checks pasaron exitosamente!")
        return 0
    else:
        print("\n Algunos checks fallaron")
        return 1

if __name__ == "__main__":
    exit(main())
