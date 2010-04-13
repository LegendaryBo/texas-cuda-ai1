package pl.wroc.uni.ii.evolution.distribution.strategies;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvReceiverImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopologyAssigner;
import pl.wroc.uni.ii.evolution.distribution.workers.EvBlankEvolInterface;
import pl.wroc.uni.ii.evolution.testhelper.EvGoalFunction;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;

public class EvImporterTest extends MockObjectTestCase {
  
  private Mock db_gateway;
  private Mock topology;
  private EvReceiverImpl<EvStunt> importer;
  private long[] neighbours = new long[] {2L,3L};;
  private List<EvStunt> expected_individuals;
  private final long task_id = 4L;
  
  @SuppressWarnings("unchecked")
  @Override
  protected void setUp() throws Exception {
    db_gateway = mock ( EvIndividualsExchanger.class );
    topology = mock (EvTopologyAssigner.class);
    importer = new EvReceiverImpl<EvStunt>(
              (EvIndividualsExchanger)db_gateway.proxy(),
              (EvTopologyAssigner)topology.proxy(),
              3, 10, task_id, 100, true);

    importer.init(new EvBlankEvolInterface());
    expected_individuals = EvStunt.list(1,2);
  }

  private EvStunt anIndividual(float val) {
    EvStunt indiv = new EvStunt(val);
    indiv.setObjectiveFunction(new EvGoalFunction());
    return indiv;
  }
  
  
  public void testImport() throws Exception {
    topology.expects(once()).method("getNeighbours").will(returnValue(neighbours));
    db_gateway.expects(once()).method("importIndividuals").with(eq(neighbours), eq(task_id),ANYTHING).will(returnValue(expected_individuals));
    importer.importIndividuals();
    assertEquals(set(expected_individuals),set(importer.getIndividuals()));
    
    assertEquals("should be queue",Collections.EMPTY_LIST,importer.getIndividuals());
  }
  
  private Set<EvStunt> set(List<EvStunt> list) {
    return new TreeSet<EvStunt>(list);
  }
  
  @SuppressWarnings("unchecked")
  public void testImporterHoldsFiniteNumberOfIndividuals() throws Exception {
    
    topology.expects(once()).method("getNeighbours").will(returnValue(neighbours));
    db_gateway.expects(once()).method("importIndividuals").with(eq(neighbours), eq(task_id),ANYTHING).will(returnValue(expected_individuals));
    importer.importIndividuals();
    
    List<EvStunt> expected_individuals2 =
      Arrays.asList(new EvStunt[] {anIndividual(3), anIndividual(4)});
    
    topology.expects(once()).method("getNeighbours").will(returnValue(neighbours));
    db_gateway.expects(once()).method("importIndividuals").with(eq(neighbours),  eq(task_id),ANYTHING).will(returnValue(expected_individuals2));
    importer.importIndividuals();
    
    List<EvStunt> result = importer.getIndividuals();
    assertEquals(3,result.size());
    
    Set<EvStunt> expected = EvStunt.set(2,3,4);
    
    assertEquals(expected,new TreeSet(result));
    assertEquals("should be queue",Collections.EMPTY_LIST,importer.getIndividuals());
  }
  
  
  @SuppressWarnings("unchecked")
  public void testImportDoesntAllowDuplicates() throws Exception {
    
    Mock db_gateway = mock (EvIndividualsExchanger.class);
    Mock topology = mock (EvTopologyAssigner.class);
    
    List<EvStunt> first_fetch = EvStunt.list(1, 2, 3, 4, 5);
    List<EvStunt> second_fetch = EvStunt.list(3, 5, 11);
    List<EvStunt> third_fetch = EvStunt.list(11, 20, 3);
    
    db_gateway.expects(exactly(3)).method("importIndividuals").with(eq(neighbours), eq(task_id), ANYTHING).will(
        onConsecutiveCalls(returnValue(first_fetch), returnValue(second_fetch), returnValue(third_fetch)));
    topology.expects(exactly(3)).method("getNeighbours").will(returnValue(neighbours));
    
    EvReceiverImpl<EvStunt> importer = new EvReceiverImpl<EvStunt>( (EvIndividualsExchanger)db_gateway.proxy(), (EvTopologyAssigner) topology.proxy(),
        100, 10, task_id, 10,  false); 
    
    importer.init(new EvBlankEvolInterface());
    importer.importIndividuals();
    importer.importIndividuals();
    importer.importIndividuals();
    List<EvStunt> result =  importer.getIndividuals();
    
    Set<EvStunt> set = new TreeSet<EvStunt>();
    set.addAll(first_fetch);
    set.addAll(second_fetch);
    set.addAll(third_fetch);
    
    assertEquals(set.size(), result.size());
    
    assertEquals(set, new TreeSet<EvStunt>(result));
    
  }
  
  
  
  
  

  @SuppressWarnings({ "unchecked", "deprecation" })
  public void testRun()  throws Exception {
    topology.expects(this.atLeastOnce()).method("getNeighbours").will(returnValue(neighbours));
    db_gateway.expects(this.atLeastOnce()).method("importIndividuals").with(eq(neighbours), eq(task_id),ANYTHING).will(returnValue(expected_individuals));
    
    importer =  new EvReceiverImpl<EvStunt>(
        (EvIndividualsExchanger)db_gateway.proxy(),
        (EvTopologyAssigner)topology.proxy(),
        3, 10, task_id, 100, false);
    
    importer.init(new EvBlankEvolInterface());
    Thread some_thread = importer.start();
    Thread.sleep(200);
    assertEquals(set(expected_individuals), set(importer.getIndividuals()));
    some_thread.stop();
  }
  
}
