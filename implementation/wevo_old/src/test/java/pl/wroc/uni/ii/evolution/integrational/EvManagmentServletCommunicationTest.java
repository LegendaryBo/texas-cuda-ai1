package pl.wroc.uni.ii.evolution.integrational;

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvTaskInfo;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.managment.communication.EvManagmentServletCommunicationImpl;

public class EvManagmentServletCommunicationTest extends TestCase {
  private static final String DISTRIBUTION_MANAGEMENT_SERVLET_ADDRESS = "http://127.0.0.1:8080/wevo_system/DistributionManager";

	public void testDummy() {
		
	} 
//  
//  public void testSampleCooperation() throws Exception {
//    EvManagmentServletCommunication proxy = new EvManagmentServletCommunicationImpl(DISTRIBUTION_MANAGEMENT_SERVLET_ADDRESS);
//
//    try {
//      long node_id = proxy.getNodeID();
//      proxy.getTaskID(node_id);
//
//      Thread.sleep(100);
//      for (int i = 0; i < 10; i++) { 
//        proxy.keepAlive(node_id);
//        Thread.sleep(50);
//      }
//    } catch(Exception ex) {
//      if (!ex.getMessage().equals("Task list for system is empty")) {
//        throw ex;
//      }
//    }
//
//  }
//
//
//  public void testaddTaskForSystem() throws Exception { 
//    EvManagmentServletCommunicationImpl proxy = new EvManagmentServletCommunicationImpl(DISTRIBUTION_MANAGEMENT_SERVLET_ADDRESS);
//
//    Random rand = new Random();
//    byte[] tab = new byte[1024];
//    rand.nextBytes(tab); 
//
//    int[] ids = proxy.getEvTaskIds();
//
//    int id1 = proxy.addTask(tab, "TASK2");
//
//    assertEquals(ids.length + 1, proxy.getEvTaskIds().length);
//
//    EvTaskInfo info = proxy.getEvTask(id1, true);
//    assertEquals(1, info.getStatus());
//
//    assertEquals(id1, info.getId());
//    assertEquals("TASK2", info.getDescription());
//
//    byte[] res = (byte[]) info.getJar();
//
//    assertEquals(tab.length, res.length);
//
//    for (int i = 0; i < tab.length; i++) {
//      res[i] = tab[i];
//    }
//
//    proxy.deleteTask(id1); 
//  }
//
//  public void testDeleteAll() throws Exception {
//    EvManagmentServletCommunicationImpl proxy = new EvManagmentServletCommunicationImpl(DISTRIBUTION_MANAGEMENT_SERVLET_ADDRESS);
//
//    proxy.addTask(new byte[] {1}, "Hello");
//
//    assertTrue(proxy.getEvTaskIds().length >= 1);
//
//    for(int id: proxy.getEvTaskIds()) {
//      proxy.deleteTask(id);
//    }
//    assertEquals(0,proxy.getEvTaskIds().length);
//
//  }
//
//  public void testAddGetStopGetResumeGet() throws Exception {
//    EvManagmentServletCommunicationImpl proxy = new EvManagmentServletCommunicationImpl(DISTRIBUTION_MANAGEMENT_SERVLET_ADDRESS);
//
//    /** delete all task */
//    for(int id: proxy.getEvTaskIds()) {
//      proxy.deleteTask(id);
//    }
//
//    /** add sample task */
//    byte[] file_task = new byte[] {1, 2, 3, 4};
//    int id1 = proxy.addTask(file_task, "COS");
//
//    /** get */
//    long node_id = proxy.getNodeID();
//
//    int id2 = proxy.getTaskID(node_id);
//
//    assertEquals(id1, id2);
//    id2 = proxy.getTaskID(node_id);
//    /** stop */
//    proxy.stopTask(id2);
//    try {
//      node_id = proxy.getNodeID();
//      proxy.getTaskID(node_id);
//      fail();
//    } catch(Exception ex) {
//
//    }
//    assertEquals(1, proxy.getEvTaskIds().length);
//
//    /** resume */
//    proxy.resumeTask(id2);
//    assertEquals(1, proxy.getEvTaskIds().length);
//
//    proxy.getTaskID(proxy.getNodeID());
//
//
//    proxy.deleteTask(id2);
//
//  }
//
//  public void testAddDelete() throws Exception {
//    EvManagmentServletCommunicationImpl proxy = new EvManagmentServletCommunicationImpl(DISTRIBUTION_MANAGEMENT_SERVLET_ADDRESS);
//
//    /** add sample task */
//    byte[] file_task = new byte[] {1, 2, 3, 4};
//    int id1 = proxy.addTask(file_task, "COS");
//
//    boolean found = false;
//    for(int id: proxy.getEvTaskIds()) {
//      if (id == id1) {
//        found = true;
//      }
//    }
//
//    int count = proxy.getEvTaskIds().length;
//    assertTrue(found);
//    assertTrue(proxy.getEvTask(id1, false) != null);
//    proxy.deleteTask(id1);
//
//    found = false;
//    for(int id: proxy.getEvTaskIds()) {
//      if (id == id1) {
//        found = true;
//      }
//    }
//    assertFalse(found);
//
//    assertTrue(count - 1 == proxy.getEvTaskIds().length);
//
//
//    /** delete all task */
//    for(int id: proxy.getEvTaskIds()) {
//      proxy.deleteTask(id);
//    }
//
//  }
//
//  public void testGetNodesCount() throws Exception {
//    EvManagmentServletCommunicationImpl proxy = new EvManagmentServletCommunicationImpl(DISTRIBUTION_MANAGEMENT_SERVLET_ADDRESS);
//
//    /** add sample task */
//    byte[] file_task = new byte[] {1, 2, 3, 4};
//    int id1 = proxy.addTask(file_task, "COS");
//
//    proxy.getNodeCountForTask(id1);
//
//
//  }


}
