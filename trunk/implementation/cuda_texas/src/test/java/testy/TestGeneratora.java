package testy;

import generator.GeneratorRozdan;
import junit.framework.TestCase;

public class TestGeneratora extends TestCase {

  public void testGeneratora() {
    
    GeneratorRozdan rozdanie = new GeneratorRozdan(13);
    GeneratorRozdan rozdanie2 = new GeneratorRozdan(13);
    assertEquals(rozdanie.toString(), rozdanie2.toString());
    System.out.println(rozdanie);
    
  }
  
}
