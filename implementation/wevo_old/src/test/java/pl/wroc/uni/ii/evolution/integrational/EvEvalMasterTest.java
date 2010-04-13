package pl.wroc.uni.ii.evolution.integrational;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.servlets.masterslave.communication.EvMasterSlaveCommunicationImpl;

public class EvEvalMasterTest extends TestCase {
  
  private String servlet_url = "http://127.0.0.1:8080/wevo_eval/EvalMaster";

//  private EvMasterSlaveCommunicationImpl master;
  

  @Override
  protected void setUp() throws Exception {
   
//    this.master = new EvMasterSlaveCommunicationImpl(servlet_url);
  }
  
  public void testOne() throws Exception {
    
//    assertTrue(master.registerComputaionNode(3, 12345, 1.23));
    //EvWorkInfo info = master.getWork(3, 12345);
    
    //System.out.println(info.work_id);
    
    
    
  }
  
  
  /*
  public void testOne() throws Exception {

    int task_id = 444;    
 
    assertFalse(master.registerComputaionNode(task_id, 12345, 4.0));


    long work_id = master.addWork(task_id, new int[] {1, 2, 3, 4, 5, 6, 7});
    EvWorkInfo work = master.getWork(task_id, 12345);
  
    assertEquals(7, work.ids.length);
    assertEquals(work_id, work.work_id);
    assertTrue(master.deleteWork(task_id, work_id));
    assertTrue(master.unregisterComputationNode(task_id, 12345));
    assertFalse(master.unregisterComputationNode(task_id, 12345));
  }
  
  
  public void testTwo() throws Exception {
    
    
    assertTrue(master.registerComputaionNode(666, 2, 30.0));
    assertTrue(master.registerComputaionNode(666, 3, 1.0));
   
    long work_id1 = master.addWork(666, new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
     
    long works[] = master.getWorks();
    assertEquals(1, works.length);
   
    
    
    
    
    EvWorkInfo info = master.getWork(666, 2);
    EvWorkInfo info2= master.getWork(666, 3);
    
    System.out.println(info.ids.length);
    System.out.println(info2.ids.length);
   
    
    
    //assertTrue(master.informWorkDone(info));
    
    assertTrue(master.informWorkDone(info2));
    info = master.getWork(666, 3);
    System.out.println(info.ids.length);
    
    assertTrue(master.informWorkDone(info));
    

    assertTrue(master.deleteWork(666, work_id1));    
    assertTrue(master.unregisterComputationNode(666, 2));
    assertTrue(master.unregisterComputationNode(666, 3));

  }*/
}

