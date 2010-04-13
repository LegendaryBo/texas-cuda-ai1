package pl.wroc.uni.ii.evolution.distribution.strategies;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.EvIndividualsExchanger;
import pl.wroc.uni.ii.evolution.distribution.strategies.exchange.ExSenderImpl;
import pl.wroc.uni.ii.evolution.distribution.strategies.topologies.EvTopologyAssigner;
import pl.wroc.uni.ii.evolution.distribution.workers.EvBlankEvolInterface;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.testhelper.EvStunt;

public class EvExporterTest extends MockObjectTestCase {
  
//  private Mock db_gateway;
//  private Mock topology;
//  private ExSenderImpl<EvStunt> exporter;
//  private List<EvStunt> four_individuals;
//  @SuppressWarnings("unused")
//  private List<EvStunt> two_individuals;
//  
//  final long signature = 1L;
//  
//  final long task_id = 3L;
//  final long client_id = 4L;
//  final long delay = 2000;
//  
//  @SuppressWarnings("unchecked")
//  @Override
//  protected void setUp() throws Exception {
//    db_gateway = mock ( EvIndividualsExchanger.class );
//    topology = mock (EvTopologyAssigner.class);
//    exporter = new ExSenderImpl<EvStunt>(
//              (EvIndividualsExchanger)db_gateway.proxy(),
//              (EvTopologyAssigner)topology.proxy(), 10, delay, task_id, client_id, 10, true);
//
//    exporter.init(new EvBlankEvolInterface());
//    four_individuals = EvStunt.list(4, 3, 2, 1);
//    two_individuals = EvStunt.list(2, 1);
//  }
//
//  /*
//  private EvStunt anIndividual(float val) {
//    EvStunt indiv = new EvStunt(val);
//    indiv.setObjectiveFunction(new EvGoalFunction());
//    return indiv;
//  }
//  */
//  public void testExport() throws Exception {
//    
//    topology.expects(once()).method("assignCellID").will(returnValue(signature));
//    db_gateway.expects(once()).method("exportIndividuals").with(eq(task_id), eq(signature), eq(client_id), eq(four_individuals));
//    exporter.export(four_individuals);
//    exporter.sendSomeIndividuals();
//  }
//  
//  /*
//  private Set<EvStunt> set(List<EvStunt> list) {
//    return new TreeSet<EvStunt>(list);
//  }
//  */
//
//  @SuppressWarnings("deprecation")
//  public void testRun() throws Exception {
//
//    topology.expects(once()).method("assignCellID")
//        .will(returnValue(signature));
//    db_gateway.expects(once()).method("exportIndividuals").with(
//       eq(task_id), eq(signature), eq(client_id), eq(four_individuals));
//    
//    Thread some_thread = exporter.start();
//    exporter.export(four_individuals);
//
//    Thread.sleep(200);
//    some_thread.stop();
//  }
//  
//  
// @SuppressWarnings("unchecked")
//public void testRemoveDuplicates() throws Exception {
//   EvBinaryVectorIndividual b1 = new EvBinaryVectorIndividual(new int[] {1, 1, 1});
//   b1.setObjectiveFunction(new OneMax());
//   
//   EvBinaryVectorIndividual b2 = new EvBinaryVectorIndividual(new int[] {0, 0, 0});
//   b2.setObjectiveFunction(new OneMax());
//  
//   EvBinaryVectorIndividual b3 = new EvBinaryVectorIndividual(new int[] {0, 1, 0});
//   b3.setObjectiveFunction(new OneMax());
//   
//   EvBinaryVectorIndividual b4 = new EvBinaryVectorIndividual(new int[] {1, 1, 1});
//   b3.setObjectiveFunction(new OneMax());
//   
//   
//   List<EvBinaryVectorIndividual> list = new ArrayList<EvBinaryVectorIndividual>();
//   list.add(b1);
//   list.add(b2);
//   list.add(b3);
//   list.add(b4);
//   
//   
//   List<EvBinaryVectorIndividual> expected = new ArrayList<EvBinaryVectorIndividual>();
//   expected.add(b1);
//   expected.add(b3);
//   expected.add(b2);
//   
//   
//   
//   
//   db_gateway.expects(once()).method("exportIndividuals").with(eq(task_id), eq(signature), eq(client_id), eq(expected));
//
//   topology.expects(once()).method("assignCellID")
//       .will(returnValue(signature));
//   ExSenderImpl exp = new ExSenderImpl<EvBinaryVectorIndividual>( (EvIndividualsExchanger)db_gateway.proxy(),
//       (EvTopologyAssigner)topology.proxy(), 4, delay, task_id, client_id, 4, false);
//  
//   exp.init(new EvBlankEvolInterface());
//   exp.export(list);
//   exp.sendSomeIndividuals();
//   
// }   
//  
// 
// @SuppressWarnings("unchecked")
//public void testSendOne() throws IOException {
//   
//   List<EvStunt> one = EvStunt.list(10);
//   List<EvStunt> two = EvStunt.list(10, 9);
//   
//   db_gateway.expects(once()).method("exportIndividuals").with(eq(task_id), eq(signature), eq(client_id), eq(one));
//
//   topology.expects(once()).method("assignCellID")
//       .will(returnValue(signature));
//   ExSenderImpl exp = new ExSenderImpl<EvBinaryVectorIndividual>( (EvIndividualsExchanger)db_gateway.proxy(),
//       (EvTopologyAssigner)topology.proxy(), 4, delay, task_id, client_id, 1, false);
//  
//   exp.init(new EvBlankEvolInterface());
//   exp.export(two);
//   exp.sendSomeIndividuals();
// }
// 
//  
// @SuppressWarnings("unchecked")
//public void testSendAll() throws IOException {
//   
//   List<EvStunt> one = EvStunt.list(10);
//   List<EvStunt> two = EvStunt.list(10, 9);
//   
//   db_gateway.expects(once()).method("exportIndividuals").with(eq(task_id), eq(signature), eq(client_id), eq(one));
//
//   topology.expects(exactly(4)).method("assignCellID")
//       .will(returnValue(signature));
//   ExSenderImpl exp = new ExSenderImpl<EvBinaryVectorIndividual>( (EvIndividualsExchanger)db_gateway.proxy(),
//       (EvTopologyAssigner)topology.proxy(), 4, delay, task_id, client_id, 1, false);
//  
//   exp.init(new EvBlankEvolInterface());
//   exp.export(two);
//   exp.sendSomeIndividuals();
//   db_gateway.expects(once()).method("exportIndividuals").with(eq(task_id), eq(signature), eq(client_id), eq(EvStunt.list(9)));
//   exp.sendSomeIndividuals();
//   db_gateway.expects(once()).method("exportIndividuals").with(eq(task_id), eq(signature), eq(client_id), eq(new ArrayList<EvStunt>()));
//   exp.sendSomeIndividuals();
//   db_gateway.expects(once()).method("exportIndividuals").with(eq(task_id), eq(signature), eq(client_id), eq(new ArrayList<EvStunt>()));
//   exp.sendSomeIndividuals();
// }
//  
//}
//class OneMax implements EvObjectiveFunction<EvBinaryVectorIndividual> {
//
//  /**
//   * 
//   */
//  private static final long serialVersionUID = -6068761739786006517L;
//
//  public double evaluate(EvBinaryVectorIndividual individual) {
//    int result = 0;
//    for (int i = 0; i < individual.getDimension(); i++) {
//      if (individual.getGene(i) == 1) {
//        result += 1;
//      }
//    }
//    return result;
//  }

	public void testDummy() {
		
	}
	
}