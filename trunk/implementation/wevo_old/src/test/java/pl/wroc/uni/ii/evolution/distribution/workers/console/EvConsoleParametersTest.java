package pl.wroc.uni.ii.evolution.distribution.workers.console;

import junit.framework.TestCase;

/**
 * 2 pharse tests.
 * 
 * @author Kacper Gorski (admin@34all.org)
 *
 */
public class EvConsoleParametersTest extends TestCase {

  
  public void testFirst() {
    String[] args = new String[]{"-a", "gra slowek", "-b", "sra polglowek",
        "-c", "stoj", "-C", "Halina", "-z", "?", "-Z", "Stalina"};
    
    EvConsoleParameters parameters = new EvConsoleParameters(args);
    
    assertEquals(parameters.getParameter('a'), "gra slowek");
    assertEquals(parameters.getParameter('b'), "sra polglowek");
    assertEquals(parameters.getParameter('y'), null);
    assertEquals(parameters.parameterExist('z'), true);
    assertEquals(parameters.parameterExist('y'), false);
    assertEquals(parameters.getParameter('z'), "?");
  }
  
  public void testSecond() {
    String[] args = new String[]{"-t", "kuszacy", "Witas", "-?",
        "Hanna z wujami", "boj w hucie", "-C", "buty jak sanie"};   
    
    EvConsoleParameters parameters = new EvConsoleParameters(args);
    
    assertEquals(parameters.getParameter('?'), null);
    assertEquals(parameters.getParameter('C'), "buty jak sanie");
    assertEquals(parameters.getParameter('t'), "kuszacy");
    
  }
  
}
