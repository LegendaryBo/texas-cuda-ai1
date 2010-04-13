package pl.wroc.uni.ii.evolution.engine.operators.general.replacement;

import java.util.TreeSet;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.replacement.EvEliteReplacement;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;

public class EliteReplacementTest extends TestCase {
  
  public void testReplacing() throws Exception {
    EvEliteReplacement<EvStunt> replecament = new EvEliteReplacement<EvStunt>(3,1);
    
    EvPopulation<EvStunt> res_pop = replecament.apply(EvStunt.pop(5,4), EvStunt.pop(1,2,3));
    
    assertEquals(EvStunt.set(5,2,3), new TreeSet<EvStunt>(res_pop));
  }
}
