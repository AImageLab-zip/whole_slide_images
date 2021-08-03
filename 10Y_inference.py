# from 10Y_main import NefroNet
cst_imp = __import__('10Y_main', fromlist=['NefroNet'])
NefroNet = cst_imp.NefroNet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', default='resnet101')
    parser.add_argument('--patches_per_bio', type=int, default=8, help='number of epochs to train')
    parser.add_argument('--preprocess', default='random', choices=['random', 'crop', 'whole_patch', 'big_whole_patch', 'glomeruli', 'big_glomeruli'])
    parser.add_argument('--classes', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--load_epoch', type=int, default=0, help='load pretrained models')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size during the training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, default=41, help='number of epochs to train')
    parser.add_argument('--SRV', action='store_true', help='is training on remote server')
    parser.add_argument('--weighted', action='store_true', help='add class weights')
    parser.add_argument('--job_id', type=str, default='', help='slurm job ID')
    parser.add_argument('--n_reps', type=int, default=20, help='number of inference reps')
    parser.add_argument('--DA', action='store_true', help='use DA during inference')

    opt = parser.parse_args()
    print(opt)

    n = NefroNet(net=opt.network, input_patches=opt.patches_per_bio, preprocess_type=opt.preprocess, num_classes=opt.classes, num_epochs=opt.epochs, batch_size=opt.batch_size,
                 l_r=opt.learning_rate, n_workers=opt.workers, job_id=opt.job_id, weights=opt.weighted)
    n.load()
    n.inference(d_loader=None, DA=opt.DA, n_reps=opt.n_reps)
