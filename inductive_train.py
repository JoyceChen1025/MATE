import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation import eval_edge_prediction
from MATE import MATE
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_inductive_data

torch.manual_seed(0)


parser = argparse.ArgumentParser('MATE self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. citHepTh or Movielens)',
                    default='citHepTh')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=15, help='Number of neighbors to sample')
parser.add_argument('--n_epoch', type=int, default=150, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true', default=True, 
                    help='Whether to augment the model with a node memory')
parser.add_argument('--memory_dim', type=int, default=128, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = args.use_memory
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_inductive_data(DATA, randomize_features=args.randomize_features)

train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

full_ngh_finder = get_neighbor_finder(full_data, args.uniform)


train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations)

device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)


n_runs_auc = []
n_runs_ap = []
n_runs_acc = []
for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  mate = MATE(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER, use_memory=USE_MEMORY, 
            memory_dimension=MEMORY_DIM, n_neighbors=NUM_NEIGHBORS)
  criterion = torch.nn.BCELoss()
  static_loss = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(mate.parameters(), lr=LEARNING_RATE)
  mate = mate.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    
    if USE_MEMORY:
      mate.memory.__init_memory__()

    mate.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        mate = mate.train()
        pos_prob, neg_prob, decode_node, raw_node_feature = mate.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(pos_prob.view(-1), pos_label) + criterion(neg_prob.view(-1), neg_label)
        loss += static_loss(decode_node, raw_node_feature)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())


      if USE_MEMORY:
        mate.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    mate.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      train_memory_backup = mate.memory.backup_memory()

    val_ap, val_auc, val_acc = eval_edge_prediction(model=mate,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS)
    if USE_MEMORY:
      val_memory_backup = mate.memory.backup_memory()
      mate.memory.restore_memory(train_memory_backup)

    nn_val_ap, nn_val_auc, nn_val_acc = eval_edge_prediction(model=mate,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      mate.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, val ap: {}, val acc: {}'.format(val_auc, val_ap, val_acc))
    logger.info(
      'new node val auc: {}, new node val ap: {}, new node val acc: {}'.format(nn_val_auc, nn_val_ap, nn_val_acc))


    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      mate.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      mate.eval()
      break
    else:
      torch.save(mate.state_dict(), get_checkpoint_path(epoch))


  if USE_MEMORY:
    val_memory_backup = mate.memory.backup_memory()

  mate.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_acc = eval_edge_prediction(model=mate,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    mate.memory.restore_memory(val_memory_backup)

  nn_test_ap, nn_test_auc, nn_test_acc = eval_edge_prediction(model=mate,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)
  n_runs_auc.append(nn_test_auc)
  n_runs_ap.append(nn_test_ap)
  n_runs_acc.append(nn_test_acc)


  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}, acc: {}'.format(nn_test_auc, nn_test_ap, nn_test_acc))
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving MATE model')
  if USE_MEMORY:
    mate.memory.restore_memory(val_memory_backup)
  torch.save(mate.state_dict(), MODEL_SAVE_PATH)
  logger.info('MATE model saved')

print('n_runs_auc:')
print(n_runs_auc)
print('n_runs_ap:')
print(n_runs_ap)
print('n_runs_acc:')
print(n_runs_acc)
logger.info(
      'Test statistics: -- New nodes -- auc: {:.2f}({:.2f}), ap: {:.2f}({:.2f}), acc: {:.2f}({:.2f})'.format(np.mean(n_runs_auc)*100, np.std(n_runs_auc)*100, np.mean(n_runs_ap)*100, np.std(n_runs_ap)*100, np.mean(n_runs_acc)*100, np.std(n_runs_acc)*100))