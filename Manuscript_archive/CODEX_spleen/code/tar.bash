tar -czvf spatial-clust-scripts-backup.tar.gz ./spatial-clust-scripts/

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Tar command was successful."
else
    echo "Tar command failed."
fi