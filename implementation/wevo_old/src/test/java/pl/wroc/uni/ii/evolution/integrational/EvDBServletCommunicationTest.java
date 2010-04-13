package pl.wroc.uni.ii.evolution.integrational;

import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.statistic.maxavgmin.EvObjectiveFunctionValueMaxAvgMinStatistic;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.objectivefunctions.binaryvector.EvOneMax;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class EvDBServletCommunicationTest extends TestCase {
  
	public void testDummy() {
		
	}	
	
//  private String WEVO_SERVER_ADDRESS = "http://127.0.0.1:8080";
//
//
//  private EvDBServletCommunication db;
//
//  private long task_id = 66666L;
//
//  private long cell_id = 0L;
//
//  private long node_id = 123L;
//
//  @Override
//  protected void setUp() throws Exception {
//    db = new EvDBServletCommunicationImpl(WEVO_SERVER_ADDRESS);
//  }
//
//  public void testAddIndividual() throws IOException {
//    EvBinaryVectorIndividual ind = new EvBinaryVectorIndividual(
//        new int[] { 1, 0 });
//
//    int id1 = db.addIndividual(ind, task_id, 1.0, cell_id, node_id);
//
//    assertTrue(id1 != -1);
//
//    EvBinaryVectorIndividual copy = (EvBinaryVectorIndividual) db.getIndividualInfo(id1,
//        true).getIndividual();
//    assertTrue(copy.equals(ind));
//
//    db.deleteIndividual(id1);
//  }
//
//  public void testDeleteIndividual() throws IOException {
//    EvBinaryVectorIndividual ind = new EvBinaryVectorIndividual(
//        new int[] { 1, 0 });
//
//    int count = db.getIndividualCount(task_id);
//    int id1 = db.addIndividual(ind, task_id, 1.0, cell_id, node_id);
//    assertEquals(count + 1, db.getIndividualCount(task_id));
//    db.deleteIndividual(id1);
//    assertEquals(count, db.getIndividualCount(task_id));
//  }
//
//  public void testDeleteIndividualFromTask() throws IOException {
//    EvBinaryVectorIndividual ind1 = new EvBinaryVectorIndividual(new int[] { 1,
//        0, 0 });
//    ind1.setObjectiveFunction(new OneMax());
//
//    EvBinaryVectorIndividual ind2 = new EvBinaryVectorIndividual(new int[] { 0,
//        0, 0 });
//    ind2.setObjectiveFunction(new OneMax());
//
//    EvBinaryVectorIndividual ind3 = new EvBinaryVectorIndividual(new int[] { 1,
//        1, 0 });
//    ind3.setObjectiveFunction(new OneMax());
//
//    db.addIndividual(ind1, task_id, 1.0, cell_id, node_id);
//    db.addIndividual(ind2, task_id, 0.0, cell_id, node_id);
//    db.addIndividual(ind3, task_id, 3.0, cell_id, node_id);
//
//    assertTrue(db.getIndividualCount(task_id) != 0);
//    db.deleteIndividualsFromTask(task_id);
//
//    assertEquals(0, db.getIndividualCount(task_id));
//
//  }
//
//  public void testAddLoadSolutionSpace() throws IOException {
//
//    OneMax objective_function = new OneMax();
//
//    EvBinaryVectorSpace space1 = new EvBinaryVectorSpace(objective_function, 13);
//    EvBinaryVectorSpace space2 = new EvBinaryVectorSpace(objective_function, 15);
//    EvBinaryVectorSpace space3 = new EvBinaryVectorSpace(objective_function, 17);
//
//    db.addSolutionSpace(task_id, 1L, space1);
//    db.addSolutionSpace(task_id, 1L, space2);
//    db.addSolutionSpace(task_id, 1L, space3);
//
//    EvBinaryVectorSpace spacef = (EvBinaryVectorSpace) db.getSolutionSpace(task_id, 1L);
//
//    assertEquals(space3.getDimension(), spacef.getDimension());
//
//  }
//
//  public void testGetBest() throws IOException {
//    EvBinaryVectorIndividual ind1 = new EvBinaryVectorIndividual(new int[] { 1,
//        0, 0 });
//    ind1.setObjectiveFunction(new OneMax());
//
//    EvBinaryVectorIndividual ind2 = new EvBinaryVectorIndividual(new int[] { 0,
//        0, 0 });
//    ind2.setObjectiveFunction(new OneMax());
//
//    EvBinaryVectorIndividual ind3 = new EvBinaryVectorIndividual(new int[] { 1,
//        1, 0 });
//    ind3.setObjectiveFunction(new OneMax());
//
//    // all individual in set
//    Set<EvIndividual> expected_set1 = new TreeSet<EvIndividual>();
//    expected_set1.add(ind1);
//    expected_set1.add(ind2);
//    expected_set1.add(ind3);
//
//    // two best individiaul in set
//
//    Set<EvIndividual> expected_set2 = new TreeSet<EvIndividual>();
//    expected_set2.add(ind1);
//    expected_set2.add(ind3);
//
//    // best individual in set
//
//    Set<EvIndividual> expected_set3 = new TreeSet<EvIndividual>();
//    expected_set3.add(ind3);
//
//    db.deleteIndividualsFromTask(task_id);
//    assertEquals(0, db.getIndividualCount(task_id));
//    int id1 = db.addIndividual(ind1, task_id, 1.0, cell_id, node_id);
//    int id2 = db.addIndividual(ind2, task_id, 0.0, cell_id, node_id);
//    int id3 = db.addIndividual(ind3, task_id, 2.0, cell_id, node_id);
//
//    assertEquals(expected_set1, toSet(db.getBestIndividualInfos(task_id, 1, 3,
//        true)));
//    assertEquals(expected_set1, toSet(db.getBestIndividualInfos(task_id, 1, 10,
//        true)));
//    // assertEquals(expected_set1,
//    // toSet(db.getBestIndividualInfosMatchingNode(task_id, node_id, 1, 3,
//    // true)));
//    assertEquals(expected_set1, toSet(db.getBestIndividualInfosMatchingCell(
//        task_id, cell_id, 1, 3, true)));
//
//    assertEquals(expected_set3, toSet(db.getBestIndividualInfos(task_id, 1, 1,
//        true)));
//    assertEquals(expected_set3, toSet(db.getBestIndividualInfosMatchingNode(
//        task_id, node_id, 1, 1, true)));
//    assertEquals(expected_set3, toSet(db.getBestIndividualInfosMatchingCell(
//        task_id, cell_id, 1, 1, true)));
//
//    assertEquals(expected_set2, toSet(db.getBestIndividualInfos(task_id, 1, 2,
//        true)));
//    assertEquals(expected_set2, toSet(db.getBestIndividualInfosMatchingNode(
//        task_id, node_id, 1, 2, true)));
//    assertEquals(expected_set2, toSet(db.getBestIndividualInfosMatchingCell(
//        task_id, cell_id, 1, 2, true)));
//
//    db.deleteIndividual(id1);
//    db.deleteIndividual(id2);
//    db.deleteIndividual(id3);
//  }
//
//  public void testGetNewVersionNumber() throws IOException {
//
//    OneMax objective_function = new OneMax();
//
//    EvBinaryVectorSpace space1 = new EvBinaryVectorSpace(objective_function, 13);
//    EvBinaryVectorSpace space2 = new EvBinaryVectorSpace(objective_function, 15);
//    EvBinaryVectorSpace space3 = new EvBinaryVectorSpace(objective_function, 17);
//
//    int ver1c = db.addSolutionSpace(task_id, 2L, space1);
//    int ver1 = db.getVersionOfNewSolutonSpace(task_id, 2L);
//
//    assertEquals(ver1c, ver1);
//
//    db.addSolutionSpace(task_id, 3L, space2);
//    assertEquals(ver1, db.getVersionOfNewSolutonSpace(task_id, 2L));
//    int c = db.addSolutionSpace(task_id, 2L, space3);
//    int c1 = db.getVersionOfNewSolutonSpace(task_id, 2L);
//    assertEquals(c, c1);
//    assertTrue(db.getVersionOfNewSolutonSpace(task_id, 2L) > ver1);
//
//  }
//
//  public void testNewSolutionSpaceAvailable() throws IOException {
//
//    EvBinaryVectorSpace space1 = new EvBinaryVectorSpace(new OneMax(), 13);
//    int ver = db.addSolutionSpace(task_id, cell_id, space1);
//    assertEquals(ver, db.getVersionOfNewSolutonSpace(task_id, cell_id));
//    db.addSolutionSpace(task_id, cell_id, space1);
//    assertTrue(db.getVersionOfNewSolutonSpace(task_id, cell_id) > ver);
//  }
//
//  public void testdeleteSolutionSpaces() throws IOException {
//
//    EvBinaryVectorSpace space1 = new EvBinaryVectorSpace(new OneMax(), 13);
//    db.addSolutionSpace(task_id, cell_id, space1);
//    boolean done = db.deleteSolutionSpaces(task_id);
//    assertEquals(0, db.getVersionOfNewSolutonSpace(task_id, cell_id));
//    int v1 = db.addSolutionSpace(task_id, cell_id, space1);
//    assertEquals(v1, db.getVersionOfNewSolutonSpace(task_id, cell_id));
//    assertTrue(v1 != 0);
//    db.deleteSolutionSpaces(task_id);
//    assertEquals(0, db.getVersionOfNewSolutonSpace(task_id, cell_id));
//    assertTrue(done);
//  }
//
//  public void testgetTaskIDS() throws IOException {
//
//    int id1 = db.addIndividual(new EvBinaryVectorIndividual(3), 1L, 2.0, 2L, 344L);
//    int id2 = db.addIndividual(new EvBinaryVectorIndividual(3), 2L, 2.0, 2L, 344L);
//    int id3 = db.addIndividual(new EvBinaryVectorIndividual(3), 11L, 2.0, 2L, 344L);
//
//    Long[] task_ids = db.getTaskIDs();
//
//    TreeSet<Long> set1 = new TreeSet<Long>(Arrays.asList(task_ids));
//
//    TreeSet<Long> set2 = new TreeSet<Long>(Arrays.asList(new Long[] { 1L, 2L,
//        11L }));
//
//    assertTrue(set1.containsAll(set2));
//
//    db.deleteIndividual(id1);
//    db.deleteIndividual(id2);
//    db.deleteIndividual(id3);
//
//  }
//
//  public void testUploadTask() throws IOException {
//
//    byte[] values = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
//    int id = db.addTaskForSystem(values, "some_tab_of_bytes");
//    EvTaskInfo info = db.getTaskForSystem(id);
//
//    assertEquals("some_tab_of_bytes", info.getDescription());
//
//    byte[] result = (byte[]) info.getJar();
//    assertEquals(values.length, result.length);
//
//    for (int i = 0; i < values.length; i++) {
//      assertEquals(values[i], result[i]);
//    }
//  }
//
//  public void testUploadBiggerTask() throws IOException {
//    byte[] file = new byte[1024 * 400];
//
//    for (int i = 0; i < file.length; i++) {
//      file[i] = (byte) (i % 1024);
//    }
//
//    int id = db.addTaskForSystem(file, "ole");
//
//   
//    EvTaskInfo info = db.getTaskForSystem(id);
//
//    byte[] file_f = (byte[]) info.getJar();
//
//    for (int i = 0; i < file.length; i++) {
//      assertEquals(file[i], file_f[i]);
//    }
//  }
//
//  public void testCombinedDeleteTaskFromSystemAndAdd() throws IOException {
//    db.addTaskForSystem(new byte[] { 1, 2 }, "test");
//    Integer[] ids = db.getTaskIDsForSystem();
//    for (Integer id : ids) {
//      db.deleteTaskFromSystem(id);
//    }
//    assertTrue(null == db.getTaskIDsForSystem());
//    int id = db.addTaskForSystem(new byte[] { 1, 2 }, "a");
//    assertEquals(1, db.getTaskIDsForSystem().length);
//    db.deleteTaskFromSystem(id);
//    assertEquals(null, db.getTaskForSystem(id));
//  }
//
//  public void testChange() throws IOException {
//    int id1 = db.addTaskForSystem(new byte[] { 1, 2 }, "test");
//    db.changeTaskState(id1, 13);
//    assertEquals(13, db.getTaskForSystem(id1).getStatus());
//    db.deleteTaskFromSystem(id1);
//  }
//  
//  public void testAddResource() throws IOException {
//    
//    
//    
//    
//    byte[] res = new byte[1024*1024];
//    Random rand = new Random();
//    rand.nextBytes(res);
//    
//    db.setResource(res, "ZDJECIE");
//    
//  
//    byte[] res_db = (byte[]) db.getResource("ZDJECIE");
//    
//    
//    if (res_db == null) {
//      System.out.println("NOTE!!! Wevo database server FAILED probably because query is to large" +
//            "\n MAX_PACKET_SIZE must be bugger!");
//    }
//    
//    assertEquals(res_db.length, res.length);
//    
//    for (int i = 0; i < res_db.length; i++) {
//      assertEquals(res[i], res_db[i]);
//    }
//
//    db.deleteResource("ZDJECIE");
//    
//    assertEquals(null, db.getResource("ZDJECIE"));
//    
//  }
//  
//  public void testAddTwoResources() throws IOException {
//     
//    db.setResource(new byte[] {1,2}, "one");
//    db.setResource(new byte[] {1,2}, "two");
//    
//    String[] names = db.getResourceNames();
//    
//    assertEquals(2, names.length);
//    db.deleteResource("one");
//    db.deleteResource("two");
//   
//  }
//
//  
//  public void testMyClassAsResorce() throws IOException {
//    
//    SameClass testclass = new SameClass();
//    testclass.val = 123;
//    
//    db.setResource(testclass, "siemka");
//    
//    SameClass result = (SameClass) db.getResource("siemka");
//    
//    assertEquals(testclass.val, result.val);
//    
//    db.deleteResource("siemka");
//    
//  }
//  
//  
//  public void testaddOneStats() throws IOException {
//    
//    long task_id = 11;
//    long cell_id = 12;
//    long node_id = 13;
//    
//    
//    EvObjectiveFunctionValueMaxAvgMinStatistic stat 
//    = new EvObjectiveFunctionValueMaxAvgMinStatistic(10, 1.0, 0.3, -1.0, -0.2);
//    
//    db.deleteStatisticForTask(task_id);
//    db.saveStatistic(task_id, cell_id, node_id, stat, 1);
//    
//    Object[] stats = db.getStatistics(task_id, cell_id, node_id);
//    assertEquals(1, stats.length);
//
//    
//    EvObjectiveFunctionValueMaxAvgMinStatistic stat_from_database = 
//     (EvObjectiveFunctionValueMaxAvgMinStatistic) stats[0];
//    
//    assertEquals(10, stat_from_database.getIteration());
//    assertEquals(1.0, stat_from_database.getMax());
//    assertEquals(0.3, stat_from_database.getAvg());
//    assertEquals(-1.0, stat_from_database.getMin());
//    db.deleteStatisticForTask(task_id);
//    
//  }
//  
//  public void testGetNodesWithStats() throws IOException {
//    
//    long task_id = 155;
//    long cell_id = 11;
//    long node_id = 14;
//   
//    EvObjectiveFunctionValueMaxAvgMinStatistic stat 
//    = new EvObjectiveFunctionValueMaxAvgMinStatistic(10, 1.0, 0.3, -1.0, 0.0);
//    
//    db.saveStatistic(task_id, cell_id, node_id, stat, 1);
//    
//    assertEquals((long)cell_id, (long)db.getCellIdsWithStatistics(task_id)[0]);
//    assertEquals((long)node_id, (long)db.getNodesIdsWithStatistics(task_id, cell_id)[0]);
//    
//    
//    db.deleteStatisticForTask(task_id);
//  }
//  
//  public void testaddNStats() throws IOException {
//    
//    long task_id = 11;
//    long cell_id = 12;
//    long node_id = 13;
//    
//    int n = 20;
//    
//    
//    EvObjectiveFunctionValueMaxAvgMinStatistic stat 
//    = new EvObjectiveFunctionValueMaxAvgMinStatistic(10, 1.0, 0.3, -1.0, 0);
//    
//    db.deleteStatisticForTask(task_id);
//    
//    for (int i = 0; i < n; i++) {
//      db.saveStatistic(task_id, cell_id, node_id, stat, 1);
//    }
//    
//    //Object[] stats = db.getStatistics(task_id, cell_id, node_id);
//    //assertEquals(n, stats.length);
//    
//    //db.deleteStatisticForTask(task_id);
//    
//  }
//  
//  public void testAddNIndividuals() throws IOException {
//   
//    int n = 30;
//    EvBinaryVectorIndividual[] ind = new EvBinaryVectorIndividual[n];
//    double[] values = new double[n]; 
//    
//    for (int i = 0; i < n; i++) {
//      ind[i] = new EvBinaryVectorIndividual(100);
//      values[i] = i;
//    }
//    
//    
//    db.deleteIndividualsFromTask(task_id);
//    db.addIndividuals(ind, task_id, values, cell_id, node_id);
//    
//    int c = db.getIndividualCount(task_id);
//    
//    assertEquals(n, c);
//    
//    db.deleteIndividualsFromTask(task_id);
//    
//  }
//  
//  
//  public void testEvalAdd() throws Exception {
//
//    int t = 555;
//    long cell = 1;
//    long node = 556;
//    int iter = 55;
//
//    
//    EvBinaryVectorIndividual ind1 = new EvBinaryVectorIndividual(10);
//    EvBinaryVectorIndividual ind2 = new EvBinaryVectorIndividual(10);
//    EvBinaryVectorIndividual ind3 = new EvBinaryVectorIndividual(10);
//    EvBinaryVectorIndividual ind4 = new EvBinaryVectorIndividual(10);
//    
//    
//    db.deleteIndividualsToEval(t);
//    int[] ids =  db.addIndividualsToEval(t, cell, node, iter,
//        new Object[] { ind1, ind2, ind3, ind4});
//    assertEquals(4, ids.length);
//    db.addIndividualsValues(ids, new double[] { 1.0, 2.0, 3.0, 4.0});
//    Object[] objs = db.getIndividualsToEval(ids);
//    assertEquals(4, objs.length);  
//    db.deleteIndividualsFromTask(t);
//
//  }
//  
//  public void testEvalAddGet() throws Exception {
//    
//    int t = 555;
//    long cell = 1;
//    long node = 556;
//    int iter = 55;
//    
//    EvBinaryVectorIndividual ind1 = new EvBinaryVectorIndividual(10);
//    EvBinaryVectorIndividual ind2 = new EvBinaryVectorIndividual(10);
//    EvBinaryVectorIndividual ind3 = new EvBinaryVectorIndividual(10);
//    EvBinaryVectorIndividual ind4 = new EvBinaryVectorIndividual(10);
//    
//    
//    
//    db.deleteIndividualsToEval(t);
//    
//    Object[] r = db.getIndividualsToEvalByIteration(t, cell, node, 1);
//    assertEquals(0, r.length);
//   
//    int[] ids = db.addIndividualsToEval(t, cell, node, iter,
//        new Object[] { ind1, ind2, ind3, ind4});
//    assertEquals(4, ids.length);
//   
//    Object[] s = db.getIndividualsToEvalByIteration(t, cell, node, 55);
//    assertEquals(4, s.length);
//    
//    Object[] k = db.getIndividualsToEvalByIteration(t, cell, node, 2);
//    assertEquals(0, k.length);
//    
//    db.deleteIndividualsToEval(t);
//    
//  }
//  
//  public void testFunAddAndGet() throws Exception {
//    
//    
//    EvOneMax fun = new EvOneMax();
//    
//    db.deleteFun(111);
//    assertFalse(db.presentFun(111));
//    db.addFun(111, fun);
//    assertTrue(db.presentFun(111));
//    Object o_fun = db.getFun(111);
//    assertNotNull(o_fun);
//    //EvOneMax fun2 = (EvOneMax) o_fun;  
//    db.deleteFun(111);
//    
//  }
//  
//  
//  
//  
//  
//  private Set<EvIndividual> toSet(EvIndividualInfo[] infos) {
//
//    TreeSet<EvIndividual> some_set = new TreeSet<EvIndividual>();
//
//    for (EvIndividualInfo info : infos) {
//      some_set.add((EvIndividual) info.getIndividual());
//    }
//
//    return some_set;
//  }
//
//}
//
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

}

class SameClass implements Serializable {
  
  /**
   * 
   */
  private static final long serialVersionUID = -136819607173243607L;
  public int val;
  public SameClass() {
    
  }
}

