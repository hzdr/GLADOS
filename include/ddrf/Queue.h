/*
 * Queue.h
 *
 *  Created on: 12.11.2015
 *      Author: Jan Stephan
 *
 *      Thread-safe queue for sharing objects.
 */

#ifndef DDRF_QUEUE_H_
#define DDRF_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace ddrf
{
	template <class Object>
	class Queue
	{
		public:
			/*
			 * Item and Object are of the same type but we need this extra template to make use of the
			 * nice reference collapsing rules
			 */
			template <class Item>
			void push(Item&& item)
			{
				auto lock = std::unique_lock<decltype(mutex_)>{mutex_};
				queue_.push(std::forward<Item>(item));
				cv_.notify_one();
			}

			Object take()
			{
				auto lock = std::unique_lock<decltype(mutex_)>{mutex_};
				while(queue_.empty())
					cv_.wait(lock);

				auto ret = std::move(queue_.front());
				queue_.pop();
				return ret;
			}

		private:
			std::mutex mutex_;
			std::condition_variable cv_;
			std::queue<Object> queue_;

	};
}

#endif /* DDRF_QUEUE_H_ */
