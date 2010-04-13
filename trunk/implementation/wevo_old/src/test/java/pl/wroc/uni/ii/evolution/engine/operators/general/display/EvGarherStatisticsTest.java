package pl.wroc.uni.ii.evolution.engine.operators.general.display;

import java.io.StringWriter;
import java.util.regex.Pattern;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.operators.general.display.EvGatherStatistics;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;

public class EvGarherStatisticsTest extends TestCase {
  
  public void testLastX() throws Exception {
    EvPopulation<EvStunt> pop = EvStunt.pop(4,6,8);
    EvGatherStatistics<EvStunt> stats = new EvGatherStatistics<EvStunt>();
    stats.apply(pop);
    assertEquals(6.0,stats.lastMean());
    assertEquals(4.0,stats.lastMin());
    assertEquals(8.0,stats.lastMax());
    assertEquals(1.62,stats.lastStD(),0.1);
  }
  
  public void testLastXonSubsequentCall() throws Exception {
    EvPopulation<EvStunt> pop1 = EvStunt.pop(1,2,3);
    EvPopulation<EvStunt> pop2 = EvStunt.pop(4,6,8);
    EvGatherStatistics<EvStunt> stats = new EvGatherStatistics<EvStunt>();
    stats.apply(pop1);
    stats.apply(pop2);
    assertEquals(6.0,stats.lastMean());
    assertEquals(4.0,stats.lastMin());
    assertEquals(8.0,stats.lastMax());
    assertEquals(1.62,stats.lastStD(),0.1);
  }
  
  public void testQueryingBeforeGatheredStats() {
    EvGatherStatistics stats = createSomeStats();
    assertEquals(0.0,stats.lastMean());
    assertEquals(0.0,stats.lastMax());
    assertEquals(0.0,stats.lastMin());
    assertEquals(0.0,stats.lastStD());
  }

  
  public void testWriteDataForGNUPLOT() throws Exception {
    StringWriter writer = new StringWriter();
    EvPopulation<EvStunt> pop = EvStunt.pop(4,6,8);
    EvGatherStatistics<EvStunt> stats = new EvGatherStatistics<EvStunt>();
    stats.apply(pop);
    stats.apply(EvStunt.pop(3,2,1));
    
    stats.writeForGnuplot(writer);
    String result = writer.toString();
    assertTrue(Pattern.matches(
        "1.0\t8.0\t6.0\t4.0\t1.63[0-9]*\t\n2.0\t3.0\t2.0\t1.0\t0.81[0-9]*\t\n", 
        result));
  }
  
  @SuppressWarnings("unchecked")
  private EvGatherStatistics createSomeStats() {
    EvGatherStatistics stats = new EvGatherStatistics();
    return stats;
  }
}
