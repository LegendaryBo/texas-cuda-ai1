package pl.wroc.uni.ii.evolution.objectivefunctions;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvSAT;

/**
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvSATTest extends TestCase {
  
  public void testConstructor() {
    int formula[][] = new int[2][];
    formula[0] = new int[] {1, -2, 3, 4, 10};
    formula[1] = new int[] {-1, 2, -12};
    
    // Check simple format
    EvSAT sat1 = new EvSAT("1 -2 3 4 10\n-1 2 -12");
    // Check cnf format
    EvSAT sat2 = new EvSAT("c test comment\np cnf 12 2\n1 -2 3 4 10 0\n-1 2 -12 0");
    // Check extra spaces and lf
    EvSAT sat3 = new EvSAT("1   -2   3 4 10\n\n-1        2 -12\n");
    
    int formula_parsed1[][] = sat1.getFormula();
    int formula_parsed2[][] = sat2.getFormula();
    int formula_parsed3[][] = sat3.getFormula();
    
    for (int i = 0; i < formula.length; i++) {
      for (int j = 0; j < formula[i].length; j++) {
        assertEquals(formula[i][j],formula_parsed1[i][j]);
        assertEquals(formula[i][j],formula_parsed2[i][j]);
        assertEquals(formula[i][j],formula_parsed3[i][j]);
      }
    }
  }
  
  
  public void testGetVariablesNumber() {
    int clausules[][] = new int[2][];
    clausules[0] = new int[] {1, -2, 3, 4, 10};
    clausules[1] = new int[] {-1, 2, -12};
    
    EvSAT sat = new EvSAT(clausules);
    
    assertEquals(sat.getVariablesNumber(),12);
  }
 
  
  public void testEvaluate() {
    int clausules[][] = new int[5][];
    clausules[0] = new int[] {1, 2, 3};
    clausules[1] = new int[] {-1, -2};
    clausules[2] = new int[] {-2, -3};
    clausules[3] = new int[] {4, 5};
    clausules[4] = new int[] {-4, -5};
    
    EvSAT sat = new EvSAT(clausules);
    
    EvBinaryVectorIndividual individual3 =
      new EvBinaryVectorIndividual(new int[] {1, 1, 1, 0, 1 });
    EvBinaryVectorIndividual individual5 =
      new EvBinaryVectorIndividual(new int[] {1, 0, 1, 1, 0 });
    
    individual3.setObjectiveFunction(sat);
    individual5.setObjectiveFunction(sat);
    
    assertEquals(individual3.getObjectiveFunctionValue(), 3.0);
    assertEquals(individual5.getObjectiveFunctionValue(), 5.0);
  }
 
}